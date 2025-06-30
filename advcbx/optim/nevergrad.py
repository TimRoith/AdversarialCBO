import torch.nn as nn
import torch
import numpy as np
import nevergrad as ng
from .base import bbobjective

class apply_from_latent:
    def __init__(self, x_orig, space):
        self.x_orig = x_orig
        self.space = space
        
    def __call__(self, d, run_idx=Ellipsis, check_valid=True):
        return self.space.apply(self.x_orig[run_idx,...], d, check_valid=check_valid, run_idx=run_idx)

class bbobjective_mod():
    def __init__(self, model, y, dtype = None, device = None, **kwargs):
        self.f = bbobjective(model, y, **kwargs)
        self.device = device
        self.dtype = dtype
        self.num_calls = 0

    def __call__(self, x, **kwargs):
        x = torch.tensor(x.value, device=self.device, dtype=self.dtype)[None, None, ...]
        self.num_calls += 1
        return self.f(x).cpu().item(), x


class NevergradAttack:
    def __init__(self, 
        model, space, 
        x_orig, y,
        check_early_stopping = True,
        opt_kwargs = None,
        targeted = False,
        loss_batch_size = 8,
        verbosity = 0,
        num_workers = 16,
        batch_mode = True,
        check_interval = 28
    ):
        self.filter_keys = ['opt-name', 'max_eval', 'popsize']
        self.targeted = targeted
        self.x_orig = x_orig
        self.space = space
        self.init_opt(opt_kwargs, space, num_opts = x_orig.shape[0])
        self.losses = [bbobjective_mod(model, y[i:i+1,...], targeted = targeted, max_batch_size = loss_batch_size, 
                                   apply = apply_from_latent(x_orig[i:i+1, ...], space), dist_reg = False, device = x_orig.device, dtype = x_orig.dtype) for i in range(x_orig.shape[0])]
        self.opt_vals  = [np.inf for _ in range(x_orig.shape[0])]
        self.x         = [None   for _ in range(x_orig.shape[0])]
        self.verbosity = verbosity
        self.num_workers = num_workers
        self.batch_mode  = batch_mode
        self.model     = model
        self.y = y
        self.check_interval = check_interval

        

    def init_opt(self, opt_kwargs, space, num_opts = 1):

        if opt_kwargs['opt-name'] == 'CMA':
            self.opt_cls = ng.optimizers.ParametrizedCMA(diagonal=False)
        elif opt_kwargs['opt-name'] == 'CMA-Diagonal':
            self.opt_cls = ng.optimizers.ParametrizedCMA(diagonal=True)
        elif opt_kwargs['opt-name'] == 'DE':
            DE_keys = ['popsize']
            lkwargs = {k:v for k,v in opt_kwargs.items() if k in DE_keys}
            self.opt_cls = ng.optimizers.DifferentialEvolution(**lkwargs)
        elif opt_kwargs['opt-name'] == '1+1':
            lkwargs = {'mutation' : 'Cauchy'}
            lkwargs = {k:v for k,v in opt_kwargs.items() if k in lkwargs.keys()}
            self.opt_cls = ng.optimization.optimizerlib.ParametrizedOnePlusOne(**lkwargs)
        else:
            raise ValueError('Unknown optimizer: ' + str(opt_kwargs['opt-name']))

        budget = opt_kwargs.get('max_eval', 1000)
        self.opts = [self.opt_cls(ng.p.Array(shape=space.dim), budget, **{k:v for k,v in opt_kwargs.items() if k not in self.filter_keys}) for _ in range(num_opts)]

    def optimize(self,):
            while self.check_and_set_active_runs():
                for idx in self.active_opts_idx:
                    opt, loss = self.opts[idx], self.losses[idx]
                    x = opt.ask()
                    loss_val, x_torch = loss(x)
                    opt.tell(x, loss_val)

                    # tracking
                    self.opt_vals[idx] = loss_val
                    self.x[idx]        = x_torch

                if self.verbosity > 0:
                    self.print_state()

    def print_state(self,):
        print(30*'-')
        print('Number of evals: ' + str(self.get_num_queries()))
        print('Energy: ' + str(self.opt_vals))

    def get_best_img(self,):
        xadv = self.x_orig.clone()
        for i in range(xadv.shape[0]):
            if self.x[i] is not None:
                xadv[i,...] = self.space.apply(self.x_orig[i:i+1,...], 
                    torch.tensor(
                        self.x[i], 
                        device=self.x_orig.device, 
                        dtype=self.x_orig.dtype
                                )
                                              )
        return xadv

    def get_num_queries(self,):
        return [opt.num_ask for opt in self.opts]

    def check_and_set_active_runs(self,):
        old_active_ops_idx = getattr(self, 'active_opts_idx', list(range(len(self.opts)))).copy()
        self.active_opts_idx = []
        xadv =  self.get_best_img()
        
        for i in old_active_ops_idx:
            opt = self.opts[i]
            active = True
            if opt.num_ask >= opt.budget:
                active = False

            if not self.targeted and self.opt_vals[i] < 0:
                active = False

            if self.targeted and (opt.num_ask % self.check_interval) == 0:
                opt._num_ask += 1
                pred = nn.Softmax(-1)(self.model(xadv[i:i+1, ...]))
                prob, labels = torch.topk(pred, k=2)
                if labels[:, 0] == self.y[i]: active = False

            if active: self.active_opts_idx.append(i)
                

        return len(self.active_opts_idx) > 0

    def get_cur_energy(self,):
        return None
                
                

        
    