import torch.nn as nn
import torch
import numpy as np
from .base import bbobjective

def tensor_clamp(x, lower, upper):
    x = torch.where(x < lower, lower, x)
    x = torch.where(x > upper, upper, x)
    return x

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
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.x_orig = x_orig[0:1,...]
        self.y = y[0:1]
        self.space = space
        self.model = model
        self.sgn = 1 - 2*targeted
        self.targeted = targeted
        self.N = N
        self.M = x_orig.shape[0]
        self.sigma = sigma
        self.max_eval = max_eval
        self.eta = eta
        self.prev_g = 0.
        self.momentum = momentum
        
        # init params
        self.x = x_orig[0:1,...].clone()
        self.x_best = torch.zeros_like(x_orig)
        self.f_min = float('inf') * torch.ones((self.M,), device=device)
        self.device = device
        self.eps = self.space.eps
        
        # scheduling
        self.last_loss = []
        self.plateau_length = plateau_length
        self.plateau_drop = plateau_drop
        self.min_eta = min_eta
        self.check_early_stopping = check_early_stopping

    def optimize(self):
        self.num_f_eval = 0

        while self.num_f_eval < self.max_eval:
            u = torch.normal(0., 1., size=(self.N, *self.x.shape[-3:]), device=self.device)
            u = torch.cat([u, -1.*u], dim=0)
            self.u = u
            inp = self.x + self.sigma * u
            
            #bs = 50
            #logits = []
            #num_batches = inp.shape[0]//bs
            with torch.no_grad():
                # for i in range(num_batches+1):
                #     logits.append(self.model(inp[(i*bs):(i+1)*bs,...]))
                #     #f_eval.append(self.f(self.inp[i*bs:(i+1)*bs,...]))
                self.logits = self.model(inp) # torch.cat(logits, dim=0)
                self.f_eval = self.loss(self.logits, self.y.expand(self.N*2))
            self.num_f_eval += self.N * 2

            # compute gradient
            g = torch.sum(self.f_eval.view(-1,1,1,1) * u, dim=0)/(2 * self.N * self.sigma)
            g = self.momentum * self.prev_g + (1.0 - self.momentum) * g
            self.prev_g = g.clone()
            
            # update
            self.x -= self.eta * torch.sign(g)
            self.x = tensor_clamp(self.x, self.x_orig - self.eps,self.x_orig + self.eps)
            self.x = torch.clamp(self.x, min=0,max=1)
            
            # update eta
            self.eta_schedule(self.f_eval.mean(dim=-1))
            
            # update best
            #self.update_best()

            # print
            self.print_step()
            
            if self.check_early_stopping:
                pred = self.model(self.x).argmax(dim=-1)
                if self.targeted:
                    eq = torch.where(pred==self.y)[0]
                    if eq.numel() > 0:
                        print('Adversarial at: ' +str(eq))
            

        return self.x
    
    def eta_schedule(self, losses):
        self.last_loss.append(losses)
        self.last_loss = self.last_loss[-self.plateau_length:]
        if self.last_loss[-1] > self.last_loss[0] and len(self.last_loss) == self.plateau_length:
            print("Annealing max_lr")
            self.eta = max(self.eta / self.plateau_drop, self.min_eta)
            self.last_loss = []
    
    def print_step(self,):
        print('Number of evaluations: ' + str(self.num_f_eval))
        print('Best energy: ' + str(self.f_eval.min(dim=0)[0].item()))
        
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