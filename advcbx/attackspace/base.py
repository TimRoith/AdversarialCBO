import torch
from pprint import pformat

def tensor_clamp(x, lower, upper):
    x = torch.where(x < lower, lower, x)
    x = torch.where(x > upper, upper, x)
    return x

def identity(x, eps, dim=(-2,-1)): return x

def project_linfty(delta, eps, dim=(-2,-1)):
    return torch.clamp(delta, min=-eps, max=eps)

def project_linfty_boundary(delta, eps, dim=(-2,-1)):
    return eps * torch.sign(delta)

def project_l2(delta, eps, dim=(-2,-1)):
    return (eps/torch.clamp(torch.linalg.vector_norm(delta, dim=dim, keepdim=True), min=eps)) * delta

def ensure_img_range(img, mrange):
    return torch.clamp(img, mrange[0], mrange[1])

def ensure_no_img_range(img, mrange):
    return img

class attack_space:
    def __init__(
        self, device='cpu', 
        img_size = None, img_range=None,
        eps=0.1,
        project = 'linfty',
    ):
        self.device = device
        self.img_size  = tuple(img_size)  if img_size is not None else (1,28,28)
        
        # image range setup
        self.img_range = img_range
        self.ensure_img_range = ensure_img_range if img_range is not None else ensure_no_img_range
        
        # budget setup
        self.eps = eps
        if project == 'linfty':
            self.project = project_linfty
        elif project == 'l2':
            self.project = project_l2
        elif project == 'linfty_boundary':
            self.project = project_linfty_boundary
        elif project == 'none' or project is None:
            self.project = identity
        else:
            self.project = project
        
    def valid_delta(self, x_orig, delta, dim=(-1)):
        delta = self.project(delta, self.eps, dim=dim) # ensure epsilon bound
        if self.img_range is not None: # ensure range bounds
            return tensor_clamp(delta, self.img_range[-2] - x_orig, self.img_range[-1] - x_orig)
        return delta
    
    def valid_adv_img(self, x_orig, x_adv):
        return self.ensure_img_range(x_orig + self.valid_delta(x_orig, x_adv - x_orig), self.img_range)
    
    def project_delta(self, delta, dim=-1):
        return self.project(delta, self.eps, dim=dim)
    
    def init_x(self, x_orig, J=100, sigma=1.):
        delta = self.sample((x_orig.shape[0], J))
        return self.valid_adv_img(x_orig[:,None,...], x_orig[:,None,...] +  sigma * delta)
    
    def __repr__(self):
        return pformat(vars(self), indent=4, width=1)
        