import torch.nn as nn
import torch
import numpy as np
from .base import bbobjective

class NESattack:
    def __init__(self, 
        model, space, 
        x_orig, y,
        N=50, 
        sigma=0.001, 
        eta=0.01, 
        max_eval=1e3,
        momentum = 0.9,
        targeted = False,
        device='cpu',
        loss_batch_size = None,
        plateau_length = 5,
        plateau_drop = 2.,
        min_eta = 5e-5,
        check_early_stopping = True
    ):
        
        self.x_orig = x_orig
        self.space = space
        self.model = model
        self.f = bbobjective(
            model, y, 
            targeted = targeted,
            max_batch_size = loss_batch_size)
        self.sgn = self.f.sgn
        self.N = N
        self.M = x_orig.shape[0]
        self.sigma = sigma
        self.max_eval = max_eval
        self.eta = eta * torch.ones((self.M,*[1 for _ in range(x_orig.ndim-1)]), device=device)
        self.prev_g = None
        self.momentum = momentum
        
        # init params
        self.x = x_orig.clone()
        self.x_best = torch.zeros_like(x_orig)
        self.f_min = float('inf') * torch.ones((self.M,), device=device)
        self.device = device
        
        # scheduling
        self.last_loss = [[] for _ in range(self.M)]
        self.plateau_length = plateau_length
        self.plateau_drop = plateau_drop
        self.min_eta = min_eta
        self.check_early_stopping = check_early_stopping

    def optimize(self):
        self.num_f_eval = 0

        while self.num_f_eval < self.max_eval:
            u = torch.normal(0, 1, size=(self.M, self.N, *self.space.dim), device=self.device)
            u = torch.cat([u, -u], dim=1)
            self.inp = self.space.apply(self.x, self.sigma * u)#.view(self.M, 2 * self.N, *self.x_orig.shape[-3:])

            # evaluate objective
            self.f_eval = self.f(self.inp)
            self.num_f_eval += u.shape[1]

            # compute gradient
            g = torch.sum(self.f_eval[(...,) + 3*(None,)] * u, dim=1,keepdims=True)/(u.shape[1] * self.sigma)
            g = self.space.to_img(g)[:,0,...]
            if self.prev_g is not None:
                #g = self.momentum * self.prev_g + (1.0 - self.momentum) * g
                pass
            self.prev_g = g.clone()
            
            # update
            self.x -= self.eta * g # torch.sign(g)
            self.x = self.space.valid_adv_img(self.x_orig, self.x)
            
            # update eta
            self.eta_schedule(self.f_eval.mean(dim=-1))
            
            # update best
            self.update_best()

            # print
            self.print_step()
            
            if self.check_early_stopping:
                pred = self.model(self.x).argmax(dim=-1)
                if self.f.targeted:
                    eq = torch.where(pred==self.f.y)[0]
                    if eq.numel() > 0:
                        print('Adversarial at: ' +str(eq))
            

        return self.x
    
    def eta_schedule(self, losses):
        for i in range(self.M):
            self.last_loss[i].append(losses[i])
            self.last_loss[i] = self.last_loss[i][-self.plateau_length:]
            if self.last_loss[i][-1] > self.last_loss[i][0] and len(self.last_loss[i]) == self.plateau_length:
                print("Annealing max_lr")
                self.eta[i,...] = max(self.eta[i] / self.plateau_drop, self.min_eta)
                self.last_loss[i] = []
    
    def print_step(self,):
        print('Number of evaluations: ' + str(self.num_f_eval))
        print('Best energy: ' + str(self.f_min))
        
    def update_cur_best(self,):
        rinp = self.inp.view(self.M,-1, *self.inp.shape[-3:])
        idx_best_loc = torch.argmin(self.f_eval, dim=-1)
        self.x_best_cur = self.space.valid_adv_img(self.x_orig, rinp[torch.arange(self.M), idx_best_loc,...])
        self.f_min_cur = self.f_eval[torch.arange(self.M), idx_best_loc]
    
    def update_best(self):
        self.update_cur_best()
        run_idx = torch.where(self.f_eval.min(dim=-1)[0] < self.f_min)
        if len(run_idx[0]) > 0:
            self.x_best[run_idx[0], ...] = self.x_best_cur[run_idx[0],...]
            self.f_min[run_idx[0]] = self.f_min_cur[run_idx[0], ...]
    
    def get_best_img(self):
        return self.x_best
    
    def get_num_queries(self):
        return self.num_f_eval
        