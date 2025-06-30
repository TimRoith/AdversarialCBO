import torch
import torch.nn.functional as F
from .base import attack_space
from math import prod
from torch_dct import idct_2d
import torch.nn.utils.prune as prune


def block_order(image_size, channels, initial_size=1, stride=1):
    '''
    copied from https://github.com/cg563/simple-blackbox-attack
    '''
    
    order = torch.zeros(channels, image_size, image_size)
    total_elems = channels * initial_size * initial_size
    perm = torch.randperm(total_elems)
    order[:, :initial_size, :initial_size] = perm.view(channels, initial_size, initial_size)
    for i in range(initial_size, image_size, stride):
        num_elems = channels * (2 * stride * i + stride * stride)
        perm = torch.randperm(num_elems) + total_elems
        num_first = channels * stride * (stride + i)
        order[:, :(i+stride), i:(i+stride)] = perm[:num_first].view(channels, -1, stride)
        order[:, i:(i+stride), :i] = perm[num_first:].view(channels, stride, -1)
        total_elems += num_elems
    return order.view(1, -1).squeeze().long().sort()[1]

class dct_attack(attack_space):
    def __init__(
        self, device='cpu', eps=0.1,
        img_size=None, img_range = None,
        modes = 5,
        project = 'l2',
        linfty_bound = 0.2,
        antithetic = False,
        latent_num_c = 1,
        initial_freq = None,
        stride = None
        ):
        super().__init__(device=device, eps=eps,img_size=img_size,img_range=img_range, project=project)
        
        self.modes = modes if isinstance(modes, int) else img_size[-1]
        self.dim = (self.modes**2 * self.img_size[-3],)
        self.m = self.img_size[-1]//2
        self.true_eps = self.eps
        # self.eps *= self.img_size[-1] # would be useful for handling differnt image dims but not for the curretn setup
        self.linfty_bound = linfty_bound
        self.antithetic = antithetic
        self.latent_num_c = latent_num_c

        if self.img_size[-1] == 299:
            self.stride = 9
            self.initial_freq = 38
        elif self.img_size[-1] == 224:
            self.stride = 5
            self.initial_freq = 29
        else:
            self.stride, self.initial_freq = stride, initial_freq
        

    def to_img(self, d):
        # zero padding
        J = d.shape[0]
        delta = torch.zeros((J, *self.img_size[-3:]), device = self.device)
        delta[..., :self.modes, :self.modes] = d.view((J, self.img_size[-3], self.modes, self.modes))
        #delta[delta.abs()<0.025] = 0
        #delta = torch.sign(delta) * 0.1
        return self.project(idct_2d(delta, norm='ortho'), self.eps, dim=(-3,-2,-1))
    
    def apply(self, img, d, check_valid=True, run_idx=None):
        d = self.project_delta(d, dim=-1)
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
    
    def valid_adv_img(self, x_orig, x_adv):
        xx_adv = x_orig + self.project_delta(x_adv-x_orig, self.eps, dim=(-3,-2,-1))
        return self.ensure_img_range(xx_adv, self.img_range)
    
    def init_delta(self, bs):
        out = self.project_delta(self.eps * torch.ones(size = bs + self.dim, device=self.device).normal_(0, self.img_range[1]/2), dim=(-3,-2,-1))
        return out
    
    def project_delta(self, delta, dim=(-3,-2,-1)):
        if self.linfty_bound:
            delta = torch.clamp(delta, min = -self.linfty_bound, max = self.linfty_bound)
        return self.project(delta, self.eps, dim=dim)

    def sample_latent(self, dyn):
        return self.sample_latent_(dyn.drift.shape)

    def sample_latent_(self, s):
        A = torch.zeros(s[:2] + self.dim, device = self.device)
        if not hasattr(self, 'latent_idx') or (self.latent_i + self.latent_num_c > self.dim[0]):
            #self.latent_idx = torch.stack([torch.randperm(self.dim[0]) for _ in range(s[0] * s[1])]).reshape(s[:2] + (-1,))
            self.latent_idx = torch.stack(
                [block_order(self.img_size[-1], self.img_size[0], initial_size=self.initial_freq, stride=self.stride)
                 for _ in range(s[0] * s[1])]).reshape(s[:2] + (-1,))
            self.latent_i = 0
        
        i = self.latent_i
        a1, a2 = torch.arange(s[0])[:,None].repeat(1, s[1]), torch.arange(s[1]).repeat(s[0],1)

        b = 1 - 2* torch.bernoulli(0.5 * torch.ones((s[0], s[1], self.latent_num_c), device = self.device))
        A.view(s[:2] + (-1,))[a1[..., None], a2[..., None], self.latent_idx[a1, a2, i:(i+self.latent_num_c)]] = b
        self.latent_i += self.latent_num_c

        if self.antithetic:
            A[:, :s[1]//2, ...] = -A[:, -s[1]//2:, ...]
        return A

    def init_semi_latent(self, s):
        return self.sample_latent_(s)
    

    
    
