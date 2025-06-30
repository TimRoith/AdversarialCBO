from cbx.dynamics import CBO
import torch
import numpy as np

class AntitheticCBO(CBO):
    def __init__(self, f, **kwargs):
        super().__init__(f, **kwargs)
    
    def compute_consensus(self,):
        x = self.x[self.consensus_idx]
        x = torch.cat([x, -x], dim=1)
        energy = self.eval_f(x) # update energy
        self.consensus, energy = self._compute_consensus(
            energy, x, self.alpha[self.active_runs_idx, :]
        )
        
        self.energy[self.consensus_idx] = energy[:, :x.shape[1]//2, ...]
