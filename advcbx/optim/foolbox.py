import foolbox as fb
from foolbox.criteria import TargetedMisclassification, Misclassification

class Foolboxattack:
    def __init__(self, model, space, x_orig, y,
                 targeted = False,
                 name = 'LinfPGD',
                 max_it = 1000,
                 max_eval = 1000,
                 device = 'cpu',
                 **kwargs):
        self.x_orig = x_orig
        self.x_best = None
        self.y = y
        
        if targeted:
            self.criterion =  TargetedMisclassification(y)
        else:
            self.criterion = Misclassification(y)
            
        self.fmodel = fb.PyTorchModel(model, space.img_range, device=device)
        self.attack = fb.attacks.LinfPGD(steps=min(max_it, max_eval))
        #self.attack = fb.attacks.LinfDeepFoolAttack(steps=min(max_it, max_eval))
        self.epsilon = space.eps

    def optimize(self,):
        clipped_advs = self.attack.run(
            self.fmodel, self.x_orig, self.criterion, 
            epsilon=self.epsilon,
        )
        self.x_best = clipped_advs
        return clipped_advs
    
    def get_best_img(self,):
        return self.x_best