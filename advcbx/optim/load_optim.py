from .CBOattack import CBOattack
from .NESattack import NESattack
#from .NESattackOrig import NESattack
from .simba import SimBA
from .foolbox import Foolboxattack
from .nevergrad import NevergradAttack
from .square_attack import SquareAttack


kwargs_CBO = {
    'sigma':10, 
    'noise': 'anistropic', 
    'dt':0.1, 
    'alpha': 10,
    'N': 100,
    'resampling': True,
    'verbosity': 0,
    'check_f_dims' :False,
    'loss_batch_size':None,
    'update_wait_thresh':20,
    'verbosity': 0,
    'kappa': 1e3,
    'factor':1.1,
    'lamda':1.,
    'max_eval':1000,
    'latent':False,
    'sampler': 'normal',
    'batch_size': None,
    'ess_eta': 0.1,
    'batch_args': {},
    'track_args':{},
    'use_latent_space_noise': False,
    'scale_by_drift_latent' : False,
    'use_heavi_correction' : False
}

kwargs_Hopping_CBO = {
    'eta':10.,
    'NES_mode': False,
    'grad_optimizer': 'gd',
    'resampling': False,
    **kwargs_CBO
}

kwargs_NES = {
    'N': 50,
    'sigma': 0.001,
    'max_eval': 1000,
    'eta': 0.01,
}

kwargs_Nevergrad = {
    'opt-name': 'CMA',
    'max_eval': 1000,
    'popsize' : 50
}

kwargs_SimBA = {
}

kwargs_SquareAttack = {
}

def load_attack_opt(model, space, img, y, cfg):
    cfgo = cfg.optim
    kwargs = {}
    if cfgo.name in ['CBO', 'SCBO', 'RandomSearch', 'SignedCBO', 'PolarCBO', 'OrthoCBO', 'HoppingCBO', 'SwitchHoppingCBO', 'SemiHoppingCBO', 'AntitheticCBO',
                     'MirrorCBO', 'OPOCBO']:
        if cfgo.name in ['HoppingCBO', 'SwitchHoppingCBO', 'SemiHoppingCBO'] :
            kwargs_known = kwargs_Hopping_CBO
        else:
            kwargs_known = kwargs_CBO
        
        kwargs = {key:cfgo[key] for key in kwargs_known.keys() if hasattr(cfgo, key)}
        opt = CBOattack(model, space, img, y,
                        targeted = cfg.attack.targeted,
                        device = cfg.device,
                        name = cfgo.name,
                        **kwargs)
    
        
    elif cfgo.name == 'NES':
        opt = NESattack(model, space, img, y,
                        targeted = cfg.attack.targeted,
                        max_eval = cfgo.max_eval,
                        device = cfg.device,
                        N=cfg.optim.N,
                        eta=getattr(cfgo, 'lr', 0.01),
                        sigma=getattr(cfgo, 'sigma', 0.001),
                        )
    elif cfgo.name == 'foolbox':
        opt = Foolboxattack(model, space, img, y, device = cfg.device,targeted = cfg.attack.targeted,)
    # elif cfgo.name == 'simba':
    #     opt = SimBA(model, img, y, max_evals=cfgo.max_eval ,eps = space.eps, targeted=cfg.attack.targeted)
    elif cfgo.name == 'Nevergrad':
        kwargs = {key:cfgo[key] for key in kwargs_Nevergrad.keys() if hasattr(cfgo, key)}
        opt = NevergradAttack(model, space, img, y, targeted = cfg.attack.targeted, opt_kwargs=kwargs)
    elif cfgo.name == 'SimBA':
        kwargs = {key:cfgo[key] for key in kwargs_SimBA.keys() if hasattr(cfgo, key)}
        opt = SimBA(model, space, img, y, targeted = cfg.attack.targeted)
    elif cfgo.name == 'SquareAttack':
        kwargs = {key:cfgo[key] for key in kwargs_SquareAttack.keys() if hasattr(cfgo, key)}
        opt = SquareAttack(model, img, y, targeted = cfg.attack.targeted)
    else:
        raise ValueError('Unknown optimizer: ' + str(cfgo.name))
    return opt
        