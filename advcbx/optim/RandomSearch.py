from cbx.dynamics import CBO

class RandomSearch(CBO):
    def __init__(self, f, **kwargs) -> None:
        super().__init__(f, **kwargs)
        self.drift = self.copy(self.x)
        
    
    def inner_step(self,) -> None:
        r"""Performs one step of the random search algorithm.

        Parameters
        ----------
        None

        Returns
        -------
        None
        
        """
        #  update particle positions
        self.x += self.noise()
        
        # update energy
        self.energy = self.eval_f(self.x).cpu().numpy()