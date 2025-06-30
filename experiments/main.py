import torch
import torch.nn as nn
import numpy as np
from advcbx.models.load_model import load_model
import hydra
from omegaconf import DictConfig, OmegaConf
import advcbx.utils as ut
from advcbx.data.load_data import load_data
import advcbx.attackspace.load_attack as la
from advcbx.optim.load_optim import load_attack_opt
from advcbx.optim.base import eval_success
from logging import info
from datetime import datetime
#%% load conf

@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg : DictConfig) -> None:
    
    info('Starting Run')
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    info('Operating on device: '+ str(cfg.device))

    # file for logging
    tar_str = 'targeted' if cfg.attack.targeted else 'untargeted'
    log_file = 'log_files/' + cfg.optim.name + '-' + cfg.model.name + '-' + tar_str + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # load model
    torch.hub.set_dir(cfg.model.path)
    model = load_model(cfg).to(cfg.device)

    ut.fix_seed(42) # fix seed per experiment to ensure same images
    # load data and initalize iterator
    test_loader = load_data(cfg)
    test_iter = iter(test_loader)
    
    # init queries and print config
    queries, queries_suc = np.zeros((0,)), np.zeros((0,))
    
    info(30*'=' + '\n' + 'Configuration:' + 30*'-')
    info(OmegaConf.to_yaml(cfg))
    with open(log_file, 'a') as f: f.write(str(OmegaConf.to_yaml(cfg)))
    
    # Main loop over number of attacks
    num_att = 0
    success = 0
    opt_print = True
    norm_diffs = {2: 0, float('inf'): 0}

    while num_att < cfg.attack.num_attacks:
        # load attack space and print info
        space = la.load_space(cfg, verbosity=0)
        info(30*'=' + '\n' + 'Loaded attack space')
        info(space)
        
        # get next batch of data
        img, y = ut.get_next_batch(test_iter, model, cfg, verbosity = 0)
        info('Attacking ' + str(img.shape[0]) + ' Images, with target classes ' + str(y))
        num_cur_att = min(img.shape[0], cfg.attack.num_attacks - num_att)
        img, y = (img[:num_cur_att,...], y[:num_cur_att])
        
        # Start optimization on current data
        opt = load_attack_opt(model, space, img, y, cfg)
        info(100*'=')
        info('Loaded Optimizer')
        if opt_print: 
            info(opt)
            opt_print = False
        info('Starting Optimization')
        opt.optimize()
        info('Finished Optimization with energy: ' +str(opt.get_cur_energy()))
        
        # evaluate attack attack
        x_adv = opt.get_best_img()
        s = eval_success(model, x_adv, y, k=1, targeted=cfg.attack.targeted)
        success += s.sum().item()
        info('Local number of successful attacks: ' + str(s.sum().item()))
        info('Local success rate: ' + str(s.sum().item()/num_cur_att))
        info('Success indices: ' + str(s.cpu().detach().numpy()))
        
        # update number of queries
        q     = np.array(opt.get_num_queries())
        q_suc = q.copy()[np.where(s.to(bool))[0]]
        queries     = np.concatenate([queries, q])
        if q_suc.size > 0: queries_suc = np.concatenate([queries_suc, q_suc])

        info('Local queries (on success): '   + str(q)                  + ' (' + str(q_suc)                  + ')')
        info('Average queries (on success): ' + str(np.mean(queries))   + ' (' + str(np.mean(queries_suc))   + ')')
        info('Median queries (on success): '  + str(np.median(queries)) + ' (' + str(np.median(queries_suc)) + ')')
        
        # update number of attacks
        num_att += num_cur_att
        info('Performed ' + str(num_att) + ' attacks, out of ' + str(cfg.attack.num_attacks) + ' total')

        # update differences
        diff = img-x_adv
        for p in norm_diffs.keys():
            norm_diffs[p] += torch.linalg.vector_norm(diff, ord=p, dim=(-1, -2, -3)).sum()
            info('Average ℓ-' + str(p) +' distance: ' + str(norm_diffs[p].item() / num_att)) 
        
        # current succes
        info('Current total success rate: ' + str(success / num_att))
        info(100*'=')
        
        
    info(200*'=')
    pstrs = [
        'Performed attacks on '          + str(num_att)            + ' different images',
        'Total successful attacks: '     + str(success),
        'Percentage: '                   + str(100 * success/num_att),
        'Average queries (on success): ' + str(np.mean(queries))   + ' (' + str(np.mean(queries_suc))   + ')',
        'Median queries (on success): '  + str(np.median(queries)) + ' (' + str(np.median(queries_suc)) + ')'
    ]
    x_adv = opt.get_best_img()
    diff = img-x_adv
    for p in norm_diffs.keys():
        norm_diffs[p] += torch.linalg.vector_norm(diff, ord=p, dim=(-1, -2, -3)).sum()
        pstrs.append('Average ℓ-' + str(p) +' distance: ' + str(norm_diffs[p].item() / num_att)) 
        
    with open(log_file, 'a') as f:
        for pstr in pstrs:
            info(pstr)
            f.write(pstr)
            f.write('\n')

    
    
if __name__ == "__main__":
    run()