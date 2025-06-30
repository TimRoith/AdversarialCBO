import torch
import random
import numpy as np

def fix_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed*2)
    torch.manual_seed(seed*3)
    torch.cuda.manual_seed(seed*4)
    torch.cuda.manual_seed_all(seed*5) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def test_model_acc(model, it, device='cuda'):
    acc = s = 0
    for x,y in it:
        x, y = x.to(device), y.to(device)
        acc += (model(x).topk(1)[1].squeeze() == y).sum()
        s   += x.shape[0]
    return acc/s
    
def get_y(y, targeted=False, num_classes=10):
    if targeted:
        c = torch.nn.functional.one_hot(y, num_classes=num_classes)
        y = torch.multinomial(1.-c,num_samples=1)[:,0]
    return y

def get_corr_idx(model, x, y):
    pred = model(x).argmax(dim=1)
    return torch.where(pred == y)[0]

def get_next_batch(iterator, model, cfg, verbosity = 1):
    x, y = next(iterator)
    img, y = (x.to(cfg.device), y.to(cfg.device))

    if getattr(cfg.attack, 'only_corr', True):
        idx = get_corr_idx(model, img, y)
        img, y = (img[idx,...], y[idx,...])
    y = get_y(y, targeted=cfg.attack.targeted, num_classes=cfg.data.num_classes)
    
    if verbosity > 0:
        print('Attacking ' + str(img.shape[0]) + ' Images, with target classes ' + str(y))
    return img, y