import torch
import torch.nn as nn
import numpy as np

def apply_default(d, run_idx=None):
    return d

def margin_loss(logits, target):
    corr_prob = logits[torch.arange(logits.shape[0]), target].clone()
    logits[torch.arange(logits.shape[0]), target] = 0 
    return logits.max(dim=-1)[0] - corr_prob

class cross_entropy_margin:
    def __init__(self, margin = 10, **kwargs):
        self.CE     = nn.CrossEntropyLoss(reduction='none')
        self.margin = margin
    
    def __call__(self, logits, targets):
        ce_loss = self.CE(logits, targets)
        pred    = logits.argmax(dim=1)
        return ce_loss - self.margin * (pred == targets)

def select_loss(name = None, targeted = False):
    if name is None:
        if targeted: 
            return cross_entropy_margin(reduction='none') # nn.CrossEntropyLoss(reduction='none')
        else:
            return margin_loss
    elif name == 'CrossEntropy':
        return cross_entropy_margin(reduction='none') #nn.CrossEntropyLoss(reduction='none')
    else:
        raise ValueError('Unknown loss: ' + str(name) + ' specified!')
    

class bbobjective:
    def __init__(self, 
                 model, y, 
                 targeted = False,
                 num_classes = 10,
                 loss = None,
                 max_batch_size=None,
                 apply = None,
                 dist_reg = False,
                 reg = None
                 ):
        self.model = model
        self.targeted = targeted
        self.sgn = -1 + 2 * self.targeted
        self.num_classes = num_classes
        self.y = y
        self.loss = select_loss(name = loss, targeted = targeted)
        self.max_batch_size = max_batch_size
        self.apply = apply if apply is not None else apply_default
        
        # set active runs
        self.set_active_runs(torch.arange(y.shape[0]))
        self.reg = reg
        
    def set_active_runs(self, run_idx):
        
        if isinstance(run_idx, (int, np.integer)):
            run_idx = [run_idx]
        self.run_idx = run_idx
        self.num_runs = len(run_idx)
            
    # def init_loss(self, l, device='cpu'):
    #     L = torch.zeros(l, device = device)
    #     # if self.dist_reg:
    #     #     L = torch.linalg.vector_norm(self.apply.x_orig[self.run_idx, None, ...] - inp.view(M,J,*inp.shape[-3:]), dim=(-3,-2,-1)).view(inpbs)
    #     #     L = 0.02 * torch.sign(L) * torch.clamp(torch.abs(L) - 0.05, min=0.) # soft thresholding, should be eps here instead of 0.05!
    #     return L
    
    def set_up_input_target(self, d):
        inp = self.apply(d, run_idx=self.run_idx)
        num_particles = inp.shape[0]//self.num_runs
        inps0 = inp.shape[0]
        
        # batch sizes
        batch_size  = inps0 if self.max_batch_size is None else min(self.max_batch_size, inps0)
        num_batches = inps0//batch_size + ((inps0 % batch_size) > 0)
        target = (self.y[self.run_idx, None] * torch.ones((self.num_runs, num_particles), device=self.y.device, dtype=torch.long)).reshape(inps0)
        return inp, target, num_batches, batch_size
        
    def __call__(self, d):
        inp, target, num_batches, batch_size = self.set_up_input_target(d)
        loss = L = torch.zeros(inp.shape[0], device = d.device)
        
        for b in range(num_batches):
            idx = slice(b * batch_size, (b+1) * batch_size)
            with torch.no_grad():
                loss[idx] += self.loss(self.model(inp[idx,...]), target[idx,...])
                
        L = self.sgn * loss.reshape(self.num_runs, -1)
        
        if self.reg is not None:
            L += self.reg(d)
        return L

        
    
def eval_success(model, x_adv, y, k=1, targeted=False):
    M = x_adv.shape[0]
    success = torch.zeros((M,))
    topk = model(x_adv).topk(k)[1]
    for m in range(M):
        if targeted:
            success[m] = y[m] in topk[m]
        else:
            success[m] = y[m] not in topk[m]

    return success
    
