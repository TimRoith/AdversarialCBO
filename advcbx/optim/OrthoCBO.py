import numpy as np
from scipy.special import logsumexp
import torch
from cbx.dynamics import CBO


#%% CBO
class OrthoCBO(CBO):
    def __init__(self, f, project: float = 1, **kwargs) -> None:
        super().__init__(f, **kwargs)
        self.project = project
        self.x_half  = self.copy(self.x)
        
    
    def inner_step(self,) -> None:
        r"""Performs one step of the OrthoCBO algorithm.

        Parameters
        ----------
        None

        Returns
        -------
        None
        
        """
        self.x_old = self.copy(self.x) # save old positions
        
        # first update
        self.compute_consensus()        
        self.drift = self.x[self.particle_idx] - self.consensus
        
        # inter step
        self.s = self.sigma * self.noise()
        self.x_half[self.particle_idx] = self.x[self.particle_idx] + 0.5 * self.s
        self.compute_consensus_half()
        d_half = self.x_half[self.particle_idx] - self.consensus_half
        
        nom   = torch.sum(self.s * d_half, dim=1, keepdims=True)
        denom = torch.sum(d_half**2, dim=1, keepdims=True)
        dd    = torch.zeros_like(nom)
        idx   = (denom > 1e-9)
        dd[idx] = (nom[idx]/denom[idx])
        #idx = (denom > 0)
        #self.s[idx] += -(nom[idx]/denom[idx]) * d_half[idx]
        self.s += -dd * d_half
        
        self.x[self.particle_idx] = self.x[self.particle_idx] - self.lamda * self.dt * self.drift + self.s

    def compute_consensus_half(self,) -> None:
        r"""Updates the weighted mean of the particles.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # evaluation of objective function on batch
        
        energy = self.eval_f(self.x_half[self.consensus_idx]) # update energy
        self.consensus_half, energy = self._compute_consensus(
            energy, self.x_half[self.consensus_idx], 
            self.alpha[self.active_runs_idx, :]
        )
        # self.energy[self.consensus_idx] = energy


class OrthoCBOExpl(CBO):
    def __init__(self, f, **kwargs) -> None:
        super().__init__(f, **kwargs)
        
    
    def inner_step(self,) -> None:
        r"""Performs one step of the OrthoCBO algorithm.

        Parameters
        ----------
        None

        Returns
        -------
        None
        
        """
        self.x_old = self.copy(self.x) # save old positions
        
        # first update
        self.compute_consensus()        
        self.drift = self.x[self.particle_idx] - self.consensus
        
        # inter step
        self.s = self.sigma * self.noise()
        
        nom   = torch.sum(self.s * self.drift, dim=1, keepdims=True)
        denom = torch.sum(self.drift**2, dim=1, keepdims=True)
        dd    = torch.zeros_like(nom)
        idx   = (denom > 1e-9)
        dd[idx] = (nom[idx]/denom[idx])
        #idx = (denom > 0)
        #self.s[idx] += -(nom[idx]/denom[idx]) * d_half[idx]
        self.s += -dd * self.drift
        
        self.x[self.particle_idx] = self.x[self.particle_idx] - self.lamda * self.dt * self.drift + self.s
        