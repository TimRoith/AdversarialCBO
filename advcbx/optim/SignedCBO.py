from cbx.dynamics import CBO
import torch

class SignedCBO(CBO):
    def __init__(self, f, **kwargs):
        super().__init__(f, **kwargs)
    
    def inner_step(self,) -> None:
        r"""Performs one step of the signed CBO algorithm.

        Parameters
        ----------
        None

        Returns
        -------
        None
        
        """
        # update, consensus point, drift and energy
        self.consensus, energy = self.compute_consensus()
        max_idx = torch.argmax(self.consensus.flatten(start_dim=2), dim=2)
        self.consensus[max_idx] = 0.05 * torch.sign(self.consensus[max_idx])
        self.consensus = torch.clamp(self.consensus, max=0.05)
        self.drift = self.x[self.particle_idx] - self.consensus
        self.energy[self.consensus_idx] = energy
        
        # compute noise
        self.s = self.sigma * self.noise()
        #dx = self.lamda * torch.sign(self.drift) + self.s
        
        dx = self.lamda *  self.dt * torch.sign(self.drift) + self.sigma * self.noise()

        #  update particle positions
        self.x[self.particle_idx] = self.x[self.particle_idx] - dx
