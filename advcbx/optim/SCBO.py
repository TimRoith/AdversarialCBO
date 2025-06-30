from cbx.dynamics.cbo import CBO, cbo_update
import numpy as np
import torch
from .torch_utils import antithetic_normal_torch
from pprint import pformat


def cbo_update(drift, lamda, dt, sigma, noise):
    return -lamda * dt * drift + sigma * noise
#%% CBO
class SCBO(CBO):


    def __init__(self, f, **kwargs) -> None:
        super().__init__(f, **kwargs)
        self.energy = None
        
    def inner_step(self,):
        # compute consensus, sets self.energy and self.consensus
        self.compute_consensus()
        
        # update drift and apply drift correction
        self.drift = self.correction(self.x[self.particle_idx] - self.consensus)
        # perform cbo update step
        self.x[self.particle_idx] += -self.lamda * self.dt * self.drift

        noise = self.sigma * self.noise()

        eold = None
        S = torch.zeros(noise.shape[:2], device = noise.device)

        for s in [-1., 0., 1.]:
            e = self.eval_f(self.x[self.active_runs_idx, ...] + s * noise)
            if not eold is None:
                idxb = torch.where(e < eold)
                S[idxb[0], idxb[1]] = s
            else:
                S[:] = s
            eold = e.clone()
            
        self.energy = e.cpu().numpy()
        self.x[self.particle_idx] += S[(Ellipsis,) + (noise.ndim-2)*(None,)] * noise


    def compute_consensus(self,) -> None:
        if self.energy is None:
            energy = self.eval_f(self.x[self.consensus_idx]) # update energy
            self.energy = energy.cpu().numpy()
        else:
            energy = torch.tensor(self.energy[self.active_runs_idx, :], device = self.x.device)

        self.consensus, energy = self._compute_consensus(
            energy, self.x[self.consensus_idx], 
            self.alpha[self.active_runs_idx, :]
        )
        self.energy[self.consensus_idx] = energy
        