import cbx
import torch
import torch.nn as nn
from torch import logsumexp, exp
from cbx.dynamics import CBO, PolarCBO, CBOMemory, MirrorCBO
from cbx.utils.termination import max_eval_term
import cbx.utils.resampling as rsmp
from cbx.scheduler import scheduler
from cbx.utils.torch_utils import norm_torch, compute_consensus_torch, compute_polar_consensus_torch, standard_normal_torch, to_numpy, to_torch_dynamic
from .torch_utils import Gaussian_kernel, effective_sample_size, multiply, antithetic_normal_torch, ElasticNet, HeaviThresholding, LogBarrierBox, Box
from .base import bbobjective
from .RandomSearch import RandomSearch
from .SignedCBO import SignedCBO
from .SCBO import SCBO
from .OPOCBO import OPOCBO
#from .StrictCBO import StrictCBO
from .OrthoCBO import OrthoCBO, OrthoCBOExpl
from .HoppingCBO import HoppingCBO, SemiHoppingCBO, SwitchHoppingCBO
from .AntitheticCBO import AntitheticCBO
import numpy as np
from pprint import pformat
from .squares import init_stripes, rect_noise

    
class post_process:
    def __init__(self, space, x_orig, latent=False, wait_thresh=5, resample=False, sigma_indep=0.005):
        self.space = space
        self.x_orig = x_orig
        M = x_orig.shape[0]
        self.latent = latent
        self.resample = resample
        #self.resampling = rsmp.resampling([rsmp.consensus_stagnation(patience=5, update_thresh=1e-1)], sigma_indep=0.005)
        self.resampling = rsmp.resampling([rsmp.loss_update_resampling(wait_thresh = 50)], sigma_indep=sigma_indep)
    
    def __call__(self, dyn):
        if self.resample: self.resampling(dyn)
        #dyn.x[:, 0,...] = dyn.best_particle
        if self.latent:
            dyn.x = self.space.project_delta(dyn.x, dim=tuple(range(2, dyn.x.ndim)))
            dyn.best_particle = self.space.project_delta(dyn.best_particle, dim=tuple(range(1, dyn.x.ndim-1)))

            #dyn.x[torch.bernoulli(torch.ones_like(dyn.x) * 0.01).to(bool)] *= -1.

            if hasattr(dyn, 'y'):
                dyn.y = torch.clamp(dyn.y, min=-1e3, max=1e3)
        else:
            dyn.x = self.space.valid_adv_img(self.x_orig[:,None,...], dyn.x)

class ProjectionInfBall:
    def __init__(self, radius=1., center=0.):
        super().__init__()
        self.radius = radius    
        self.center = center
        self.thresh = 1e-5

    def grad(self, x):
        return x
        
    def grad_conj(self, y):
        return self.center + torch.clamp(y - self.center, min=-self.radius, max=self.radius)

                    
class adversarial_term:
    def __init__(self, prob_thresh=0.0, reeval = False, untargeted_loss_thresh = -0.01):
        self.prob_thresh = prob_thresh
        self.reeval      = reeval
        self.untargeted_loss_thresh = untargeted_loss_thresh
    
    def __call__(self, dyn):
        if self.reeval:
            pred = nn.Softmax(-1)(dyn.f.model(dyn.f.apply(dyn.best_particle[dyn.active_runs_idx, None,:], 
                                                          check_valid=True, run_idx=dyn.active_runs_idx)))
            prob, labels = torch.topk(pred, k=2)
            dyn.num_f_eval[dyn.active_runs_idx] += 1
        if dyn.f.targeted:
            # determine the indices where the highest probability is at least prob_thresh higher than the second one
            #cleared_prob = (prob[:, 0] - prob[:, 1]) > self.prob_thresh
            #eq = torch.logical_and(labels[:, 0]==dyn.f.y[dyn.active_runs_idx], cleared_prob)

            if not self.reeval:
                eq = dyn.best_energy[dyn.active_runs_idx] < 0.
                neg_idx = np.where(eq)[0]
                potential_adv_idx = dyn.active_runs_idx[eq]

                # check each index individually to prevent batch effects
                for j,i in enumerate(potential_adv_idx):
                    pred = dyn.f.model(
                        dyn.f.apply(
                            dyn.best_particle[i:i+1, None,:], 
                            check_valid=True, 
                            run_idx=[i]))
                    dyn.num_f_eval[i] += 1
                    prob, labels = torch.topk(pred, k=2)
                    if not (labels[:, 0] == dyn.f.y[i]):
                        eq[neg_idx[j]] = False
                
                eq = torch.tensor(eq)
            else:
                cleared_prob = (prob[:, 0] - prob[:, 1]) > self.prob_thresh
                eq = torch.logical_and(labels[:, 0]==dyn.f.y[dyn.active_runs_idx], cleared_prob)
        else:
            if self.reeval:
                eq = (labels[:, 0] !=dyn.f.y[dyn.active_runs_idx])
            else:
                eq = torch.tensor(dyn.best_energy[dyn.active_runs_idx] < self.untargeted_loss_thresh)
            
        # update active indices in objective function
        dyn.f.set_active_runs(dyn.active_runs_idx[torch.where(~eq)[0].cpu()])
        
        if dyn.verbosity > 0:
            print('Adversarial at: ' + str([i for i in range(dyn.M) if i not in dyn.active_runs_idx]))
            
        term = np.ones((dyn.M,), dtype=bool)
        term[dyn.f.run_idx] = False
        return term
    
class constant_noise:
    def __call__(self, dyn):
        return torch.normal(0.,1., size=dyn.drift.shape, device=dyn.drift.device)
    
class heavi_side_correction:
    def __call__(self, dyn, x):
        dyn.num_f_eval += dyn.consensus.shape[0] # update number of function evaluations
        return self.correct(x, dyn.f, dyn.energy[dyn.active_runs_idx,...], dyn.consensus)

    def correct(self, x, f, energy, consensus):
        z = torch.tensor(energy, device = x.device) - f(consensus)
        return x * torch.where(z > 0, 1,0)[(..., ) + (None, ) * (x.ndim-2)]
        
def get_cbx_optimizer(name, f, pp, x=None, x_orig=None, sampler=None, space = None, device='cpu', 
                      use_heavi_correction = False,
                      update_wait_thresh = 50, kappa=1e3, batch_size = None, **kwargs):
    dyn_dict = {'CBO':CBO,
                'SwitchHoppingCBO':SwitchHoppingCBO,
                'SemiHoppingCBO':SemiHoppingCBO,
                'RandomSearch':RandomSearch, 'SignedCBO':SignedCBO, 
                'OrthoCBO':OrthoCBO, #OrthoCBO, 
                'AntitheticCBO':AntitheticCBO,
                'SCBO': SCBO,
                'OPOCBO': OPOCBO}
    if name in dyn_dict.keys():
        batch_args = {} if batch_size is None else {'size':batch_size, 'partial':False}

        correction = heavi_side_correction() if use_heavi_correction else 'no_correction'
            
        
        dyn = to_torch_dynamic(dyn_dict[name])(
            f, x=x, f_dim='3D',
            norm=norm_torch,
            copy=torch.clone,
            sampler=sampler,
            compute_consensus = compute_consensus_torch,
            post_process=pp,
            batch_args=batch_args,
            correction = correction,
            **kwargs
        )
    elif name == 'HoppingCBO':
        dyn = HoppingCBO(
            f, x=x, f_dim='3D',
            norm=norm_torch,
            copy=torch.clone,
            sampler=sampler,
            compute_consensus = compute_consensus_torch,
            post_process=pp,
            **kwargs
        )
        dyn.x *= 0
    elif name == 'MirrorCBO':
        batch_args = {} if batch_size is None else {'size':batch_size, 'partial':False}
        pp.resampling.var_name = 'y'

        mm = Box(r = 0.05) # ElasticNet(lamda=0.1)
        
        dyn = to_torch_dynamic(MirrorCBO)(
            f, x=x, f_dim='3D',
            norm=norm_torch,
            copy=torch.clone,
            sampler=sampler,
            compute_consensus = compute_consensus_torch,
            post_process=pp,
            mirrormap = mm,
            batch_args=batch_args,
            #correction = heavi_side_correction(),
            **kwargs
        )
    elif name == 'PolarCBO':
        dyn = PolarCBO(
            f, x=x, f_dim='3D',
            norm=norm_torch,
            copy=torch.clone,
            sampler=sampler,
            compute_consensus=compute_polar_consensus_torch,
            post_process=pp,
            kernel='Gaussian',
            kappa = kappa,
            kernel_factor_mode='alpha',
            **kwargs
        ) 
    else:
        raise ValueError('Unknown method name: ' + str(name))
        
    return dyn


        
class track_drift:
    def init_history(self, dyn):
        dyn.history['drift_mean'] = []
    def update(self, dyn):
        dyn.history['drift_mean'].append(torch.mean(torch.abs(dyn.drift)).cpu().numpy())
    
class apply_from_latent:
    def __init__(self, x_orig, space):
        self.x_orig = x_orig
        self.space = space
        
    def __call__(self, d, run_idx=Ellipsis, check_valid=True):
        return self.space.apply(self.x_orig[run_idx,...], d, check_valid=check_valid, run_idx=run_idx)
    
class apply_image:
    def __init__(self, x_orig):
        self.x_orig = x_orig
    def __call__(self, x, run_idx=Ellipsis, check_valid=True):
        return x.flatten(end_dim=1)
    
class anisotropic_to_img:
    def __init__(self, space):
        self.space = space

    def __call__(self, dyn):
        return np.sqrt(dyn.dt) * self.sample(dyn.drift)

    def sample(self, drift):
        z = self.space.sample(drift.shape[:2])
        return z * drift
    
class isotropic_to_img:
    def __init__(self, space):
        self.space = space

    def __call__(self, dyn):
        return np.sqrt(dyn.dt) * self.sample(dyn.drift)

    def sample(self, drift):
        z = self.space.sample(drift.shape[:2])
        return z * torch.linalg.vector_norm(drift, dim=tuple(i for i in range(2,drift.ndim)), keepdims=True)
    
    
class space_noise:
    def __init__(self, space):
        self.space = space
        self.sample = self.space.sample
    def __call__(self, dyn):
        return np.sqrt(dyn.dt) * self.sample(dyn.drift.shape[:2])

class latent_space_noise:
    def __init__(self, space, scale_by_drift = True):
        self.space = space
        self.sample = self.space.sample_latent
        self.scale_by_drift = scale_by_drift
    
    def __call__(self, dyn):
        out = np.sqrt(dyn.dt) * self.sample(dyn)
        if self.scale_by_drift:
            drift = dyn.drift
            out *= torch.linalg.vector_norm(drift, dim=tuple(i for i in range(2,drift.ndim)), keepdims=True)
        return out
    
class CBOattack:
    def __init__(self, model, space, x_orig, y,
                 targeted = False,
                 N = 30,
                 resampling = False,
                 loss_batch_size = None,
                 device = 'cpu',
                 name = 'CBO',
                 factor = 1.005,
                 max_eval=1000,
                 latent = False,
                 noise = 'anisotropic',
                 update_wait_thresh = 25,
                 ess_eta = 0.1,
                 max_alpha = 1e7,
                 sampler = 'normal',
                 use_latent_space_noise = False,
                 scale_by_drift_latent  = False,
                 **kwargs):
        self.device = device
        self.factor = factor
        x_orig = x_orig.clone()
        self.space = space
        self.x_orig = x_orig
        self.latent = latent
        self.ess_eta = ess_eta
        self.max_alpha = max_alpha
        self.name = name
        sampler = self.get_sampler(sampler)

        M = x_orig.shape[0]
        
        if latent:
            # if noise == 'squares':
            #     x = space.init_delta((M, N))
            #     x = init_stripes(x)
            #     noise = rect_noise(max_calls=3000)
            # else:
            x = space.init_delta((M, N))

            if noise == 'isotropic':
                noise = cbx.noise.isotropic_noise(norm = norm_torch, sampler = sampler)
            else:
                noise = cbx.noise.anisotropic_noise(norm = norm_torch, sampler = sampler)
                
            apply = apply_from_latent(x_orig, space)
        else:
            x = space.init_x(x_orig, J=N).reshape(M,N,*x_orig.shape[-3:])
            apply = apply_image(x_orig)
            noise = space_noise(space)

        if use_latent_space_noise:
            noise = latent_space_noise(space, scale_by_drift = scale_by_drift_latent)
            x = space.init_semi_latent(x.shape)

        f = bbobjective(
            model, y,
            targeted = targeted,
            max_batch_size = loss_batch_size,
            apply = apply,
            dist_reg = False,
        )
        print(resampling)
        pp = post_process(space, x_orig, wait_thresh=update_wait_thresh, latent = latent, resample=resampling)
        self.dyn = get_cbx_optimizer(
            name, f, pp, x=x, x_orig = x_orig,
            space = space,
            device = device,
            noise = noise,
            sampler = sampler,
            term_criteria = [max_eval_term(max_eval), adversarial_term(reeval=(name in ['HoppingCBO', 'NES']))],
            **kwargs)
        
    def get_sampler(self, name):
        if name == 'normal':
            return standard_normal_torch(self.device)
        elif name == 'antithetic':
            return antithetic_normal_torch(self.device)
        else:
            raise ValueError('Unknown sampler: ' + str(name) + '. Choose from "normal" or "antithetic"')
        
    
    def optimize(self, sched = 'select', print_int = 1):
        if sched == 'select':
        
            sched = effective_sample_size(eta=self.ess_eta, maximum=self.max_alpha, device=self.x_orig.device)
            #sched = multiply(name='alpha', maximum=1e18, factor=1.005)
            if self.name == 'HoppingCBO':
                sched = None#multiply(name='alpha', maximum=1e2, factor=1.005)
        #sched = scheduler([multiply(name='alpha', maximum=1e6, factor=self.factor),])# multiply(name='dt', maximum=10., factor=0.999)])
        return self.dyn.optimize(sched=sched, print_int = print_int)
    
    def get_best_img(self,):
        if self.latent:
            return self.space.apply(self.x_orig, self.dyn.best_particle[:,None,:], check_valid=True)
        else:
            return self.dyn.f.apply(self.dyn.best_particle[:,None,:])
        
    def get_num_queries(self,):
        return self.dyn.num_f_eval
    
    def get_cur_energy(self,):
        return self.dyn.best_cur_energy
    
    def __repr__(self):
        v_dict = vars(self)
        exclude = ['x_orig']
        v_dict = {k:v for k,v in v_dict.items() if not k in exclude}
        return pformat(v_dict, indent=4, width=1)
