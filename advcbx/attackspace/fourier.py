import torch
import torch.nn.functional as F
from .base import attack_space
from math import prod
from torch.fft import irfft2

class fourier_attack(attack_space):
    def __init__(
        self, device='cpu', eps=0.1,
        img_size=None, img_range = None,
        modes = 5,
        interp_mode = 'nearest'
        ):
        super().__init__(device=device, eps=eps,img_size=img_size,img_range=img_range)
        
        self.modes = modes
        self.dim = 2*self.modes**2
        self.interp_mode = interp_mode
        self.m = self.img_size[-1]//2

    def to_img(self, d):
        # zero padding
        J = d.shape[0]
        delta = torch.zeros((J, self.img_size[-2], self.m), device = self.device)
        delta[:,self.m-self.modes:self.m+self.modes,:][...,-self.modes:] = d.view((J, 2*self.modes, self.modes))
        return irfft2(delta, dim = (-2,-1), norm='ortho', s=self.img_size[-2:])
    
    def apply(self, img, d, check_valid=True):
        if d.ndim == 2:
            d = d[:, None, :]
        M = d.shape[0]
        J = d.shape[-2]
        delta = self.to_img(d.reshape(-1,d.shape[-1])).reshape(M, J, *self.img_size)
        
        if check_valid:
            delta = self.valid_delta(img[:,None,...], delta)
        inp = delta + img[:,None,...]
        inp = self.ensure_img_range(inp, self.img_range)
        inp = inp.reshape([M*J] + list(img.shape[-3:]))
        return inp
    
    def init_delta(self, M=1, J=100):
        return torch.ones(size = (M, J, self.dim), device=self.device).uniform_(-self.eps, self.eps)
    def project_delta(self, delta):
        return self.project(delta, 10.)
    
    
