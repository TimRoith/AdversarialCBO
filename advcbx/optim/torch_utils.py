import numpy as np
import torch
from cbx.scheduler import param_update, bisection_solve, eff_sample_size_gap

def antithetic_normal_torch(device):
    def _normal_torch(size = (1,)):
        N_half = size[1]//2
        R = size[1]%2
        size_half = (size[0], N_half + R, *size[2:])
        z = torch.randn(size_half).to(device)
        return torch.cat([z, -z[:, R:, ...]], dim=1)
    return _normal_torch

class effective_sample_size(param_update):
    def __init__(self, name = 'alpha', eta=.1, maximum=1e5, solve_max_it = 100, device='cpu'):
        super().__init__(name = name, maximum=maximum)
        self.eta = eta
        self.solve_max_it = solve_max_it
        self.device = device
        
    def update(self, dyn):
        val = getattr(dyn, self.name)
        val = bisection_solve(
            eff_sample_size_gap(dyn.energy, self.eta), 
            self.minimum * np.ones((dyn.M,)), self.maximum * np.ones((dyn.M,)), 
            max_it = self.solve_max_it, thresh=1e-2
        )
        setattr(dyn, self.name, torch.tensor(val[:, None], device=self.device, dtype=torch.float32))
        #self.ensure_max(dyn)
        
class multiply:
    def __init__(self, name='alpha', maximum = 1e5, factor = 1.0,):
        self.name = name
        self.maximum = maximum
        self.factor = factor
    
    def update(self, dyn) -> None:
        old_val = getattr(dyn, self.name)
        new_val = torch.clamp(self.factor * old_val, max=self.maximum)
        #new_val = min(self.factor * old_val, self.maximum)
        setattr(dyn, self.name, new_val)
        
class variance_sched:
    def __init__(self, name='alpha', maximum = 1e5, factor = 1.0, var=1e-5):
        self.name = name
        self.maximum = maximum
        self.factor = factor
        self.var = var
        
    def update(self, dyn) -> None:
        val = getattr(dyn, self.name)
        var = torch.var(dyn.x,dim=-2).mean(axis=-1).min()
        if var < self.var:
            val *= 1/self.factor
        else:
            val *= self.factor
        setattr(dyn, self.name, torch.clamp(val, max=self.maximum))

def compute_polar_consensus_torch(energy, x, neg_log_eval, alpha = 1., kernel_factor = 1.):
    weights = -kernel_factor * neg_log_eval - alpha * energy[:,None,:]
    coeffs = torch.exp(weights - logsumexp(weights, dim=(-1,), keepdims=True))[...,None]
    c = torch.sum(x[:,None,...] * coeffs, axis=-2)
    return c, energy.cpu().numpy()

class Gaussian_kernel:
    def __init__(self, kappa = 1.0):
        self.kappa=kappa
    
    def __call__(self, x, y):
        dists = torch.linalg.norm(x-y, dim=-1)
        return torch.exp(-torch.true_divide(1, 2*self.kappa**2) *dists**2)
    
    def neg_log(self, x, y):        
        dists = torch.linalg.norm(x-y, dim=-1, ord=2)
        return torch.true_divide(1, 2*self.kappa**2) * dists**2


class HeaviThresholding:
    def __init__(self, lmin = 1.0, lmax = None):
        self.lmin = lmin
        self.lmax = lmin if lmax is None else lmax

    def grad(self, theta):
        return 4/(theta**5)

    def grad_conj(self, theta):
        #x = theta.clone()

        x = (theta/self.lmax)**5/4
        #x[theta.abs() < self.lmin] = 0
        return torch.clamp(x, min=-self.lmax, max=self.lmax)


class ElasticNet:

    def __init__(self, delta=1.0, lamda=1.0):
        self.delta = delta
        self.lamda = lamda

    def __call__(self,theta):
        return (1/(2 * self.delta))* torch.sum(theta**2, axis=-1) + self.lamda * torch.sum(torch.abs(theta), axis=-1)
    
    def grad(self, theta):
        return (1/(self.delta)) * theta + self.lamda * torch.sign(theta)
    
    def grad_conj(self, y):
        return self.delta * torch.sign(y) * torch.clamp((torch.abs(y) - self.lamda), min=0)

class LogBarrierBox:
    def __init__(self, r=1, dim=(-1,-2,-3)):
        self.r = r
        self.dim = dim
        
    def __call__(self, theta):
        return torch.sum(np.log(1/(self.r - theta)) + np.log(1/(self.r + theta)), axis = self.dim)
    
    def grad(self, theta):
        return 1/(self.r - theta) - 1/(self.r + theta)
    
    def grad_conj(self, y):
        return torch.nan_to_num((-1 + (1 + y**2 * self.r**2)**0.5) / y)

class Box:
    def __init__(self, r=1, dim=(-1,-2,-3)):
        self.r = r
        self.dim = dim
        
    def __call__(self, theta):
        return torch.sum(np.log(1/(self.r - theta)) + np.log(1/(self.r + theta)), axis = self.dim)
    
    def grad(self, theta):
        return theta
    
    def grad_conj(self, y):
        x = torch.sign(y) * self.r
        return x
    