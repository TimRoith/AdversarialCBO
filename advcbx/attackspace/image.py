import torch
import torch.nn.functional as F
from .base import attack_space
from math import prod

class image_attack(attack_space):
    def __init__(
        self, device='cpu', eps=0.1,
        img_size=None, img_range = None,
        ):
        super().__init__(device=device, eps=eps,img_size=img_size,img_range=img_range)
        
        self.dim = self.img_size

    def to_img(self, d):
        J = d.shape[-2]
        return d.view(J, *self.img_size)
    
    def apply(self, img, d):
        M = d.shape[0]
        J = d.shape[1]
        delta = d.reshape(M, J, *self.img_size)
        delta = self.valid_delta(img[:,None,...], delta)
        inp = delta + img[:,None,...]
        inp = self.ensure_img_range(inp, self.img_range)
        inp = inp.reshape([M*J] + list(img.shape[-3:]))
        return inp
    
    def init_delta(self, M=1, J=100):
        return torch.ones(size = (M, J, self.dim), device=self.device).uniform_(-self.eps, self.eps)
    
    
