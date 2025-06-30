import torch
import torch.nn.functional as F
from .base import attack_space
from math import prod

class low_res_attack(attack_space):
    def __init__(
        self, device='cpu', eps=0.1,
        img_size=None, img_range = None,
        img_lr_size = None,
        interp_mode = 'nearest'
        ):
        super().__init__(device=device, eps=eps,img_size=img_size,img_range=img_range, project='linfty')
        
        self.img_lr_size = tuple(img_lr_size) if img_lr_size is not None else (14,14)
        self.dim = self.img_lr_size
        self.interp_mode = interp_mode

    def to_img(self, d):
        return F.interpolate(d.flatten(end_dim=1), size=self.img_size[-2:], mode=self.interp_mode).view(d.shape[:2] + self.img_size)
    
    def apply(self, img, d, check_valid=True, run_idx=None):
        if d.ndim == 2:
            d = d[:, None, :]
        #d = self.eps * torch.sign(d)
        delta = self.to_img(d).view(d.shape[:2] + self.img_size)
        if check_valid:
            delta = self.valid_delta(img[:,None,...], delta)
        inp = delta + img[:,None,...]
        inp = self.ensure_img_range(inp, self.img_range)
        return inp.flatten(end_dim=1)
    
    def init_delta(self, bs):
        # return torch.ones(size = bs + self.img_lr_size, device=self.device).uniform_(-self.eps, self.eps)
        return torch.sign(torch.ones(size = bs + self.img_lr_size, device=self.device).uniform_(-1., 1.)) * self.eps
    
    def sample(self, bs = (1,5)):
        z = torch.normal(0., 1., size = bs + tuple(self.img_lr_size), device = self.device)
        return self.to_img(z)
    

class low_res_attack_tanh(low_res_attack):
    def __init__(
        self, **kwargs
        ):
        super().__init__(**kwargs)
    
    def apply(self, img, d, check_valid=True, run_idx=None):
        if d.ndim == 2:
            d = d[:, None, :]
        #d = self.eps * torch.tanh(d)
        delta = self.to_img(d).view(d.shape[:2] + self.img_size)
        delta = self.eps * torch.tanh(delta)

        inp = delta + img[:,None,...]
        inp = self.ensure_img_range(inp, self.img_range)
        return inp.flatten(end_dim=1)
    
    def project_delta(self, delta, dim=-1):
        return self.project(delta, 10, dim=dim)
    
    def init_delta(self, bs):
        return torch.ones(size = bs + self.img_lr_size, device=self.device).uniform_(-1., 1.)


class low_res_attack_discrete(low_res_attack):
    def __init__(
        self, **kwargs
        ):
        super().__init__(**kwargs)
    
    def apply(self, img, d, check_valid=True, run_idx=None):
        if d.ndim == 2:
            d = d[:, None, :]
        #d = self.eps * torch.tanh(d)
        d  = 1 - 2 * torch.bernoulli(1/(1 + torch.exp(-d)))
        delta = self.to_img(d).view(d.shape[:2] + self.img_size)
        delta = self.eps * delta

        inp = delta + img[:,None,...]
        inp = self.ensure_img_range(inp, self.img_range)
        return inp.flatten(end_dim=1)
    
    def project_delta(self, delta, dim=-1):
        return self.project(delta, 1000, dim=dim)
    
    def init_delta(self, bs):
        return torch.ones(size = bs + self.img_lr_size, device=self.device).uniform_(-10., 10.)
    
    

    
