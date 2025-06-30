import torch
from .base import attack_space

class index_attack(attack_space):
    def __init__(
        self, n=5, device='cpu', eps=1.0, 
        img_size = None, img_range= None,
        x_min=0.2, x_max=0.8
    ):
        super().__init__(device=device, eps=eps,img_size=img_size,img_range=img_range, project='linfty')

        self.n = n
        self.dim = n
        self.x_min = x_min
        self.x_max = x_max

        self.dim = (self.n, 5)

    def to_img(self, d):
        J = d.shape[0]
        delta = torch.zeros((J,) +  self.img_size, device=self.device)
        idx = (d[..., :2] * self.img_size[-1]).to(torch.int32)
        c = idx.reshape(J, -1, 2)
        idx = torch.remainder(c, self.img_size[-1])

        d_idx = torch.arange(J, device=self.device, dtype=torch.long)
        delta[d_idx[:,None], :, idx[:,:,0], idx[:,:,1]] = d[..., 2:]
        return delta

    def project_delta(self, delta, dim=-1):
        delta[..., :2] = torch.clamp(delta[..., :2], min=-5,max=5.)
        delta[..., 2:] = torch.clamp(delta[..., 2:], min=-5.,max=5.)
        return delta

    def init_delta(self, s):
        return torch.zeros(s + (self.n, 5), device=self.device).uniform_(-self.eps, self.eps)
    
    def apply(self, img, d, check_valid=True, run_idx=None):
        if d.ndim == 2: d = d[:, None, :]
        M, N = d.shape[0], d.shape[1]

        delta = self.to_img(d.reshape(-1, self.n, 5))
        delta = delta.reshape((M, N) + self.img_size)
        inp = self.ensure_img_range(delta + img[:,None,...], self.img_range)
        return inp.flatten(end_dim=1)