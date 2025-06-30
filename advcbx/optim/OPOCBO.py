import numpy as np
from scipy.special import logsumexp
import torch
from cbx.dynamics import CBO


#%% CBO
class OPOCBO(CBO):
    def __init__(self, f, antithetic = False, **kwargs) -> None:
        super().__init__(f, **kwargs)
        self.x_new  = self.copy(self.x)
        self.energy[:] = 1e10
        self.antithetic = antithetic
        
    
    def inner_step(self,) -> None:
        r"""Performs one step of the OrthoCBO algorithm.

        Parameters
        ----------
        None

        Returns
        -------
        None
        
        """
        
        self.drift = torch.zeros_like(self.x[self.consensus_idx])
        self.s     = self.sigma * self.noise()

        self.x_new[self.consensus_idx] += self.s

        energy   = self.eval_f(self.x_new[self.consensus_idx]) # update energy
        energy_old_torch = torch.tensor(self.energy[self.consensus_idx], dtype=self.x.dtype, device=self.x.device)

        improved = 1. * (energy < energy_old_torch)
        energy   = (1 - improved) * energy_old_torch + improved * energy

        if self.antithetic:
            self.x_new[self.consensus_idx] -= 2 * self.s
            energy_a    = self.eval_f(self.x_new[self.consensus_idx]) # update energy
            improved_a  = 1. * (energy_a < energy)
            improved   -= 1. * improved_a
            energy      = (1 - improved_a) * energy + improved_a * energy_a

        
        self.x[self.consensus_idx]     +=  improved[(Ellipsis,) + (self.x.ndim-2)*(None,)] * self.s
        self.energy[self.consensus_idx] = self.to_numpy(energy)

         # save old positions where the loss was evaluated for logging
        self.x_old = self.copy(self.x)

        
        self.consensus, energy = self._compute_consensus(
            energy, self.x[self.consensus_idx], 
            self.alpha[self.active_runs_idx, :]
        )
        
        # drift     
        self.drift = self.x[self.particle_idx] - self.consensus

        self.x[self.particle_idx] = self.x[self.particle_idx] - self.lamda * self.dt * self.drift
        self.x_new = self.copy(self.x)


        