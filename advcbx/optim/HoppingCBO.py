from cbx.dynamics.cbo import CBO, cbo_update
import numpy as np
import torch
from .torch_utils import antithetic_normal_torch
from pprint import pformat

class NES_gradient:
    def __init__(self, sigma):
        self.sigma = sigma
        
    def __call__(self, energy, u, alpha):
        g = torch.sum(energy[(...,) + (u.ndim-2)*(None,)] * u, dim=1, keepdims=True)/(u.shape[1])
        return -g, energy.cpu().numpy()
    
class momentum_gd:
    def __init__(self, momentum=0.9, nesterov=True):
        self.momentum = momentum
        self.b = None
        self.nesterov = nesterov
        
    def step(self, grad, idx):
        if self.b is not None:
            self.b[idx, ...] = self.momentum * self.b[idx, ...] + (1-self.momentum) * grad
        else:
            self.b = grad.clone()
            
        if self.nesterov:
            grad += self.momentum * self.b[idx, ...]
        else:
            grad = self.b[idx, ...]
            
        return grad
    
class gd:
    def step(self, grad, idx):
        return grad
    
class adam:
    def __init__(self, beta_1 = 0.9, beta_2 = 0.999, eps=1e-8):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.it = 0
        self.eps = eps
        
    def step(self, grad, idx):
        if self.it == 0:
            self.m = torch.zeros_like(grad)
            self.v = torch.zeros_like(grad)
        self.it+=1
            
        self.m[idx, ...] = self.beta_1 * self.m[idx, ...] + (1 - self.beta_1) * grad
        self.v[idx, ...] = self.beta_2 * self.v[idx, ...] + (1 - self.beta_2) * grad**2
        
        mm = self.m[idx, ...]/(1-self.beta_1**self.it)
        vv = self.v[idx, ...]/(1-self.beta_2**self.it)
        return mm/(torch.sqrt(vv) + self.eps)
    
def get_grad_optimizer(name, **kwargs):
    if name == 'gd':
        return gd()
    elif name == 'momentum_gd':
        return momentum_gd(**kwargs)
    elif name == 'adam':
        return adam(**kwargs)
    else:
        raise ValueError('Unknown Gradient Optimizer: ' + str(name) + '. ' +
                         'Please choose from gd, momentum_gd, adam')

def l2_proj(x):
    return x/torch.linalg.vector_norm(x, dim=tuple(range(2, x.ndim)), keepdims=True)



def l2_proj(x):
    return x/torch.linalg.vector_norm(x, dim=tuple(range(2, x.ndim)), keepdims=True)


def update_best_cur_particle_hopping(self,) -> None:
    
    
    self.best_cur_energy = self.energy.min(axis=1)
    self.f_min = self.best_cur_energy
    self.f_min_idx = self.energy.argmin(axis=-1)
    if not hasattr(self, 'best_cur_particle'): self.best_cur_particle = self.copy(self.x)[:, 0, ...]

    if self.best_mode == 0:
        self.best_cur_particle = self.x[:, 0, ...]
    else:
        self.best_cur_particle[self.active_runs_idx, ...] = self.eval_points[
            np.arange(self.num_active_runs), 
            self.f_min_idx[self.active_runs_idx, ...], :]

        

def update_best_particle_hopping(self,):
    idx = np.where(self.best_energy > self.best_cur_energy)[0]
    if len(idx) > 0:
        self.best_energy[idx] = self.best_cur_energy[idx]
        self.best_particle[idx, :] = self.copy(self.best_cur_particle[idx, :])
        
def eta_schedule_hopping(self, losses):
    for j, idx in enumerate(self.active_runs_idx):
        self.last_loss[idx].append(losses[j])
        self.last_loss[idx] = self.last_loss[idx][-self.eta_plateau:]
        if self.last_loss[idx][-1] > self.last_loss[idx][0] and len(self.last_loss[idx]) == self.eta_plateau:
            self.eta[idx,...] = max(self.eta[idx] / self.eta_drop, self.min_eta)
            self.last_loss[idx] = []

            if self.verbosity > 0:
                print('Annealing eta in run: ' + str(idx) + ' to ' + str(self.eta[idx].item()))
                
def hopping_step(self,):
        # compute new evaluation points by adding noise to current state x
        s = self.noise()
        self.eval_points = self.x[self.active_runs_idx, ...] + self.sigma * s
        
        # compute energy at evaluation points and perform gradient estimation
        energy         = self.eval_f(self.eval_points) # update energy
        grad, energy   = self.estimate_grad(energy, s, self.alpha[self.active_runs_idx, :])
        grad = l2_proj(grad)
        
        # perform update based on underlying optimizer
        self.x[self.active_runs_idx, ...] -= self.eta[self.active_runs_idx, :] * self.grad_optimizer.step(-1. * grad, self.active_runs_idx)
        
        # update energy and perform eta scheduling
        self.energy[self.active_runs_idx, ...] = energy
        self.eta_schedule(np.mean(energy, axis=-1))

#%% CBO
class HoppingCBO(CBO):
    def __init__(self, f, eta=1., min_eta = 5e-5, 
                 eta_drop = 1.5, eta_plateau=5,
                 sampler = None,
                 NES_mode=False, 
                 grad_optimizer='gd',
                 best_mode = 0,
                 **kwargs) -> None:
        super().__init__(f, **kwargs)
        self.x = 0 * self.copy(self.x[:,0:1,...])
        self.sampler = antithetic_normal_torch(self.x.device) if sampler is None else sampler
        self.eta = eta
        self.min_eta = min_eta
        self.eta_drop = eta_drop
        self.eta_plateau = eta_plateau
        self.last_loss = [[] for _ in range(self.M)]
        self.eta = eta * torch.ones((self.M, *[1 for _ in range(self.x.ndim-1)]), device=self.x.device)
        self.grad_optimizer = get_grad_optimizer(grad_optimizer)
        self.grad_normalizer = l2_proj
        self.grad_optimizer.step(-1. * torch.zeros_like(self.x), torch.arange(self.M))
        self.best_mode = best_mode
        
        if NES_mode:
            self.estimate_grad = NES_gradient(sigma=self.sigma)
        else:
            self.estimate_grad = self._compute_consensus
    
    def noise(self,):
        return self.sampler(size = (self.num_active_runs, self.N, *self.d))
      
        
    inner_step = hopping_step
    update_best_cur_particle = update_best_cur_particle_hopping
    update_best_particle = update_best_particle_hopping
    eta_schedule = eta_schedule_hopping
                    
    print_vars = ['grad_optimizer', 'grad_normalizer', 'eta', 'eta_drop', 'min_eta', 'eta_plateau', 'estimate_grad'] + CBO.print_vars
    



#%% Semi Hopping CBO
class SemiHoppingCBO(CBO):
    def __init__(
        self, f, eta=1., min_eta = 5e-3, 
        eta_drop = 1.5, eta_plateau=5,
        sampler = None,
        NES_mode=False, 
        grad_optimizer='gd',
        sigma_hopping = None,
        N_hopping=50,
        **kwargs) -> None:
        
        super().__init__(f, **kwargs)
        self.x[:,0,...] *= 0
        self.sampler = antithetic_normal_torch(self.x.device) if sampler is None else sampler
        self.eta = eta
        self.min_eta = min_eta
        self.eta_drop = eta_drop
        self.eta_plateau = eta_plateau
        self.last_loss = [[] for _ in range(self.M)]
        self.eta = eta * torch.ones((self.M, *[1 for _ in range(self.x.ndim-1)]), device=self.x.device)
        self.grad_optimizer = get_grad_optimizer(grad_optimizer)
        self.grad_normalizer = l2_proj
        self.sigma_hopping = 0.001 * self.sigma if sigma_hopping is None else sigma_hopping
        self.N_hopping = N_hopping
        
        if NES_mode:
            self.estimate_grad = NES_gradient(sigma=self.sigma)
        else:
            self.estimate_grad = self._compute_consensus
    
    def const_noise(self,):
        return antithetic_normal_torch(device=self.x.device)(size = (self.num_active_runs, self.N_hopping, *self.d))
    
    def hopping_step(self,):
        # comute new evaluation points by adding noise to current state x
        s = self.const_noise()        
        eval_points = self.x[self.active_runs_idx, :1, ...] + self.sigma_hopping * s
        
        # compute energy at evaluation points and perform gradient estimation
        energy         = self.eval_f(eval_points) # update energy
        grad, energy   = self.estimate_grad(energy, s, self.alpha[self.active_runs_idx, :])
        grad = l2_proj(grad)
        
        # perform update based on underlying optimizer
        self.x[self.active_runs_idx, :1, ...] -= self.eta[self.active_runs_idx, :] * self.grad_optimizer.step(-1. * grad, self.active_runs_idx)
        # update energy and perform eta scheduling
        self.eta_schedule(np.mean(energy, axis=-1))
        self.energy[self.active_runs_idx, 0, ...] = np.mean(energy, axis=-1)
        
    def cbo_step(self,):
        # compute consensus, sets self.energy and self.consensus
        self.compute_consensus()
        # update drift and apply drift correction
        self.drift = self.correction(self.x[self.particle_idx] - self.consensus)
        # perform cbo update step
        self.x[self.active_runs_idx, 1:, ...] += cbo_update(
            self.drift, self.lamda, self.dt, 
            self.sigma, self.noise()
        )[:, 1:, ...]
        
    def inner_step(self,) -> None:
        prob = min(self.it/500, .9)
        p = torch.distributions.binomial.Binomial(total_count=1, probs=torch.tensor([prob])).sample()
        # if p < 0.5:
        #     self.cbo_step()
        # else:
        self.cbo_step()
        self.hopping_step()
        
            
    def eta_schedule(self, losses):
        for j, idx in enumerate(self.active_runs_idx):
            self.last_loss[idx].append(losses[j])
            self.last_loss[idx] = self.last_loss[idx][-self.eta_plateau:]
            if self.last_loss[idx][-1] > self.last_loss[idx][0] and len(self.last_loss[idx]) == self.eta_plateau:
                self.eta[idx,...] = max(self.eta[idx] / self.eta_drop, self.min_eta)
                self.last_loss[idx] = []
                
                if self.verbosity > 0:
                    print('Annealing eta in run: ' + str(idx) + ' to ' + str(self.eta[idx].item()))
                    
    print_vars = ['grad_optimizer', 'grad_normalizer', 'eta', 'eta_drop', 'min_eta', 'eta_plateau', 'estimate_grad'] + CBO.print_vars



#%% Switch Hopping CBO
class SwitchHoppingCBO(CBO):
    def __init__(
        self, f, eta=1., min_eta = 5e-5, 
        eta_drop = 1.5, eta_plateau=5,
        sampler = None,
        NES_mode=False, 
        grad_optimizer='momentum_gd',
        sigma_hopping = 0.001,
        max_cbo_it = 100,
        alpha_hopping = 0.1,
        **kwargs) -> None:
        
        super().__init__(f, **kwargs)
        self.sampler = antithetic_normal_torch(self.x.device) if sampler is None else sampler
        self.eta = eta
        self.min_eta = min_eta
        self.eta_drop = eta_drop
        self.eta_plateau = eta_plateau
        self.last_loss = [[] for _ in range(self.M)]
        self.eta = eta * torch.ones((self.M, *[1 for _ in range(self.x.ndim-1)]), device=self.x.device)
        self.grad_optimizer = get_grad_optimizer(grad_optimizer)
        self.grad_normalizer = l2_proj
        self.sigma_hopping = sigma_hopping
        self.alpha_hopping = alpha_hopping
        self.switched = False
        self.max_cbo_it = max_cbo_it
        
        if NES_mode:
            self.estimate_grad = NES_gradient(sigma=self.sigma)
        else:
            self.estimate_grad = self._compute_consensus
            
    hopping_step = hopping_step
    eta_schedule = eta_schedule_hopping
    
    def const_noise(self,):
        return self.sampler(size = (self.num_active_runs, self.N, *self.d))
    
    def noise(self,):
        if self.switched:
            return self.const_noise()
        else:
            return self.noise_callable(self)
    
      
    def switch_to_hopping(self,):
        self.switched = True
        self.x = self.best_particle[:, None, ...]
        self.sampler = antithetic_normal_torch(self.x.device)
        self.grad_optimizer.step(torch.zeros_like(self.x), torch.arange(self.M))
        self.sigma = self.sigma_hopping
        self.N *= 2
        self.energy = np.concatenate(2*[self.energy.copy()], axis=1)
        if self.verbosity > 0:
            print(100*'=')
            print('Switched to Hopping')
        
    def inner_step(self,) -> None:
        if self.it < self.max_cbo_it:
            self.cbo_step()
        else:
            if not self.switched:
                self.switch_to_hopping()
            self.hopping_step()
            self.alpha[:] = self.alpha_hopping
            
    def update_best_cur_particle(self,):
        if self.switched:
            update_best_cur_particle_hopping(self)
        else:
            CBO.update_best_cur_particle(self,)
            
    def update_best_particle(self,):
        if self.switched:
            update_best_particle_hopping(self)
        else:
            CBO.update_best_particle(self,)
            
                    
    print_vars = ['grad_optimizer', 'grad_normalizer', 'eta', 'eta_drop', 'min_eta', 'eta_plateau', 'estimate_grad'] + CBO.print_vars


