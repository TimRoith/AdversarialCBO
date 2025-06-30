import torch
import torch.nn as nn
import numpy as np
from cbx.utils.objective_handling import cbx_objective
from .low_res import low_res_attack, low_res_attack_tanh, low_res_attack_discrete
from .image import image_attack
from .fourier import fourier_attack
from .dct import dct_attack
from .square import square_attack, square_attack_nl
from .index import index_attack

__all__ = ['load_space']

def load_space(cfg, verbosity=1):
    cfga = cfg.attack
    img_range = getattr(cfg.data, 'img_range', None)
    kwargs = {'eps': cfga.eps, 'device':cfg.device, 'img_range':img_range,
              'img_size': cfg.data.shape}
    C = cfg.data.shape[0]
    if cfga.name in ['low_res', 'low_res_tanh', 'low_res_discrete']:
        M_low = getattr(cfg.attack, 'M_low', cfg.attack.N_low)
        
        interp_mode = getattr(cfg.attack, 'interp_mode', 'nearest')
        if cfga.name=='low_res':
            at = low_res_attack
        elif cfga.name=='low_res_discrete':
            at = low_res_attack_discrete
        else:
            at = low_res_attack_tanh
        
        attack = at(
            img_lr_size = [C, cfga.N_low, M_low],
            interp_mode = interp_mode,
            **kwargs
        )
    
    elif cfga.name == 'index':
        attack = index_attack(
            n=cfga.n, 
            **kwargs
            )
    elif cfga.name == 'image':
        attack = image_attack(
            **kwargs
            )
    elif cfga.name == 'fourier':
        modes = getattr(cfg.attack, 'modes', 5)
        attack = fourier_attack(modes=modes, **kwargs)
    elif cfga.name == 'dct':
        modes = getattr(cfg.attack, 'modes', 14)
        attack = dct_attack(modes=modes, 
                            linfty_bound = getattr(cfga, 'linfty_bound', None),
                            project = getattr(cfga, 'project', 'linfty'),
                            antithetic = getattr(cfga, 'antithetic', False),
                            **kwargs)
    elif cfga.name == 'square':
        N = getattr(cfg.optim, 'N', 15)
        M = getattr(cfg.data, 'batch_size_test', 8)
        num_squares = getattr(cfga, 'num_squares', 100)
        attack = square_attack(bs=(M,N), num_squares=num_squares, **kwargs)
    elif cfga.name == 'square-nl':
        attack = square_attack_nl(mq = getattr(cfga, 'mq', 1e4),
                                  p_init = getattr(cfga, 'p_init', .1),
                                  **kwargs)
    else:
        raise ValueError('Unknown attack: ' + str(cfga.name))
    
    if verbosity > 0:
        print(30* '-' + '\n' + 'Loaded attack space: ')
        print(attack)
    return attack