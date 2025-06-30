import torch
import torch.nn.functional as F
from .base import attack_space
from math import prod
import numpy as np

def p_selection(p_init, it, n_iters):
    """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
    it = int(it / n_iters * 10000)

    if 10 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 8
    elif 500 < it <= 1000:
        p = p_init / 16
    elif 1000 < it <= 2000:
        p = p_init / 32
    elif 2000 < it <= 4000:
        p = p_init / 64
    elif 4000 < it <= 6000:
        p = p_init / 128
    elif 6000 < it <= 8000:
        p = p_init / 256
    elif 8000 < it <= 10000:
        p = p_init / 512
    else:
        p = p_init

    return p

def get_random_rect_idx(x, w = 30, fix_channel_pos = True):
    mgrid = list(torch.meshgrid(tuple([torch.arange(x.shape[i]) for i in range(3)] + 
                                      #[torch.arange(1)] + # this keeps the color channel fixed
                                      [torch.arange(w)          for i in range(2)])))
    if fix_channel_pos:
        idx = [torch.randint(0, x.shape[i] - w, x.shape[:2] + (1,)) for i in [-2, -1]]
    else:
        idx = [torch.randint(0, x.shape[i] - w, x.shape[:3]) for i in [-2, -1]]
    
    for i in [-1, -2]: 
        mgrid[i] = torch.clamp(mgrid[i] + idx[i%2][..., None, None], min=0, max=x.shape[i] - 1)

    return tuple(mgrid)
    
def rectangle_sampler(x, w = 30, f = 0.1, fix_channel_pos = True):
    mgrid = get_random_rect_idx(x, w = w, fix_channel_pos = fix_channel_pos)
    delta = torch.zeros_like(x)
    delta[mgrid] = f * (1-2 * torch.bernoulli(0.5 * torch.ones_like(delta[mgrid][...,:1,:1])))
    return delta

class rect_noise:
    def __init__(self, max_calls=1e5, p_init=0.05, im_w = 299, num_rects = 1, n_features = 3):
        self.max_calls = max_calls
        self.p_init = p_init
        self.num_calls = 0
        self.im_w = im_w
        self.num_rects  = num_rects
    
    def __call__(self, dyn):
        self.select_w(dyn.it,  dyn.x.shape[-3] * dyn.x.shape[-2] *dyn.x.shape[-1], dyn.x.shape[-3])
        out = torch.zeros_like(dyn.drift)
        
        for i in range(self.num_rects):
            out += rectangle_sampler(dyn.drift, w=self.w//(i+1), fix_channel_pos = True)
        return out


    def select_w(self, it, n_features, c):
        self.num_calls += 1 #bs//2
        #self.w = int(p_selection(self.p_init, self.num_calls, self.max_calls)**0.5 * self.im_w)

        p = p_selection(self.p_init, it, self.max_calls)
        self.w = max(int(round(np.sqrt(p * n_features / c))), 3)

class rect_noise_og:
    def __init__(self, max_calls=1e5, p_init=0.05, im_w = 299, num_rects = 1, n_features = 3):
        self.max_calls = max_calls
        self.p_init = p_init
        self.num_calls = 0
        self.im_w = im_w
        self.num_rects  = num_rects

    def __call__(self, dyn):
        out = torch.zeros_like(dyn.drift)
        p = p_selection(self.p_init, dyn.it, self.max_calls)
        c, h, w    = (dyn.x.shape[-3], dyn.x.shape[-2], dyn.x.shape[-1])
        n_features = c * h * w
        eps = 1
        
        for i_img in range(out.shape[0]):
            for n in range(out.shape[1]):
                s = int(round(np.sqrt(p * n_features / c)))
                s = min(max(s, 1), h-1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
                center_h = np.random.randint(0, h - s)
                center_w = np.random.randint(0, w - s)
    
                #x_curr_window = x_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
                #x_best_curr_window = x_best_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
                # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
                #while np.sum(np.abs(np.clip(x_curr_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s], min_val, max_val) - x_best_curr_window) < 10**-7) == c*s*s:
                out[i_img, n, :, center_h:center_h+s, center_w:center_w+s] = torch.tensor(np.random.choice([-eps, eps], size=[c, 1, 1]), device = out.device)
        return out    

class square_attack(attack_space):
    def __init__(
        self, device='cpu', eps=0.1,
        img_size=None, img_range = None,
        num_squares = 40,
        bs = None
        ):
        super().__init__(device=device, eps=eps,img_size=img_size,img_range=img_range, project='linfty')
        self.num_squares = num_squares
        

    def init_stripes(self, bs):
        # can use torch multinomial here
        self.stripes = torch.tensor(
            np.random.choice([-self.eps, self.eps], size=(bs[0], 1, self.img_size[0], 1, self.img_size[-1])), 
            device=self.device, dtype=torch.float) #
        delta = torch.zeros(size = bs + (self.num_squares, 3), device = self.device)
        delta[...,:2].uniform_(0, 1)
        delta[...,2].uniform_(0.01, 0.2)

        self.sgn = torch.sgn(torch.zeros(size = (bs[0],) + (self.num_squares, 3), device = self.device).uniform_(-1,1))
        #delta[...,3:].normal_(0.,1.)
        #x = torch.zeros(size = bs + self.img_size, device=self.device)
        #return x + torch.tensor(np.random.choice([-self.eps, self.eps], size=tuple(bs) + (self.c, 1, self.w)), device = self.device, dtype=torch.float)
        return self.project_delta(delta)

    def init_delta(self, bs):
        # can use torch multinomial here
        self.stripes = torch.tensor(
            np.random.choice([-self.eps, self.eps], size=(bs[0], 1, self.img_size[0], 1, self.img_size[-1])), 
            device=self.device, dtype=torch.float) #
        delta = torch.zeros(size = bs + (self.num_squares, 3), device = self.device)
        delta[...,:2].uniform_(0, 1)
        delta[...,2].uniform_(0.01, 0.2)
        self.sgn = torch.sgn(torch.zeros(size = (bs[0],) + (self.num_squares, 3), device = self.device).uniform_(-1,1))
        #delta[...,3:].normal_(0.,1.)
        #x = torch.zeros(size = bs + self.img_size, device=self.device)
        #return x + torch.tensor(np.random.choice([-self.eps, self.eps], size=tuple(bs) + (self.c, 1, self.w)), device = self.device, dtype=torch.float)
        return self.project_delta(delta)
    
    def apply(self, x_orig, d, check_valid = False, run_idx = Ellipsis):
        M,N = d.shape[:2]
        d = d.flatten(end_dim=1)
        
        idx = (d[..., :2] * self.img_size[-1]).to(int)
        #w   = self.img_size[-1]/(self.num_squares**0.5) * torch.ones((1,1), device=d.device)#
        w   = (d[...,  2] * self.img_size[-1]).to(int)
        #w   = (torch.linspace(0.01, 0.3, self.num_squares, device=d.device)[None, None:].repeat(M* N, 1) *
        #       self.img_size[-1]).to(int)
        
        #sgn = d[..., 3:3+self.img_size[0]]
        #sgn = torch.sign(sgn)
        #stripes = torch.sign(d[..., :self.img_size[0], (3+self.img_size[0]):]).view((M*N,self.img_size[0],1,self.img_size[-1]))
        stripes = self.stripes[run_idx, ...].repeat(1,N,1,1,1).view(N*M,self.img_size[0],1,self.img_size[-1])
        a = torch.arange(0, self.img_size[-1], device = d.device).repeat((M*N, d.shape[1],1))
        b = ((a[...,None] > idx[...,0:1,None]) & (a[...,None] < (idx[...,0:1,None] + w[...,None,None])))
        b = b & ((a[...,None,:] > idx[...,None,1:2]) & (a[...,None,:] < (idx[...,None,1:2]+ w[...,None,None])))
        b = b.view((M, N) + b.shape[1:])
        b = b[:,:,:,None,...].repeat(1,1,1,3,1,1)

        if run_idx is Ellipsis:
            sgn = self.sgn[:,None, ..., None, None]
        else:
            sgn = self.sgn[run_idx, None, ..., None, None]
        
        b = b * sgn
        b *= torch.arange(self.num_squares, device=d.device)[(None, None, Ellipsis) + 3*(None,)]**4
        
        delta = b.sum(dim=2).flatten(end_dim=1)
        delta = torch.clamp(delta, min = -self.eps, max = self.eps)
        delta += stripes * (delta.abs().sum(dim=1,keepdims=True) < 1e-10) 
        x_adv = x_orig[:,None,...] + torch.clamp(delta, min = -self.eps, max = self.eps).view((M,N, *self.img_size))
        if check_valid:
            x_adv = self.valid_adv_img(x_orig[:,None,...], x_adv)
        
        return x_adv.flatten(end_dim=1)
                
    def project_delta(self, delta, dim=-1):
        delta[...,2]  = torch.clamp(delta[...,2],  min=0.1, max = 0.5)
        #delta[...,2]  = delta[...,2]/torch.linalg.vector_norm(delta[...,2], dim=-1, keepdims=True) * self.img_size[-1]
        delta[...,:2] = torch.clamp(delta[...,:2], min=0, max=.8)              
        # delta[...,:2] = torch.where(delta[...,:2] > self.img_size[-1] - delta[...,2:3], self.img_size[-1] - delta[...,2:3], delta[...,:2])
        #delta[...,3:] = torch.clamp(delta[...,3:], min=-self.eps, max=self.eps)
        return delta
    
    def init_x(self, x, J=100, sigma=1.):
        delta = torch.tensor(np.random.choice([-self.eps, self.eps], size=[x.shape[0], J, self.c, 1, self.w]), device=self.device, dtype=x.dtype)
        return self.valid_adv_img(x[:,None,...], x[:,None,...] + delta)

    def sample(self, drift):
        bs = drift.shape[:2]
        p = p_selection(self.p_init, self.iter, self.n_iters)
        deltas = np.zeros((prod(bs),) + self.img_size)
        for i_img in range(deltas.shape[0]):
            s = int(round(np.sqrt(p * self.dim / self.c)))
            s = min(max(s, 1), self.h-1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
            
            center_h = np.random.randint(0, self.h - s)
            center_w = np.random.randint(0, self.w - s)

            #x_curr_window = x_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            #x_best_curr_window = x_best_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
            #while np.sum(np.abs(np.clip(x_curr_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s], min_val, max_val) - x_best_curr_window) < 10**-7) == c*s*s:
            deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = np.random.choice([-self.eps, self.eps], size=[self.c, 1, 1])
        self.iter += 1
        return torch.tensor(deltas, device=self.device, dtype=torch.float).view(bs + self.img_size)



class square_attack_nl(attack_space):
    def __init__(
        self, device='cpu', eps=0.1,
        img_size=None, img_range = None,
        bs = None,
        mq = 10000,
        p_init = 0.05,
        antithetic = False
        ):
        super().__init__(device=device, eps=eps,img_size=img_size,img_range=img_range, project='linfty')
        self.dim = tuple(img_size)
        self.rect_noise = rect_noise(max_calls=mq, p_init=p_init, im_w = img_size[-1], num_rects = 1)
        self.mq = mq
        self.p_init = p_init
        self.antithetic = antithetic
        
    def init_semi_latent(self, s):
        self.rect_noise = rect_noise_og(max_calls=self.mq, p_init=self.p_init, im_w = self.img_size[-1], num_rects = 1)
        out = torch.zeros(s, device = self.device)
        self.stripes = self.eps * (1-2 * torch.bernoulli(0.5 * torch.ones_like(out[..., :1, :])))
        return out + self.stripes

    def init_delta(self, bs):
        return torch.zeros(bs + self.dim, device = self.device)

    def apply(self, img, delta, check_valid = False, run_idx = Ellipsis):
        if check_valid:
            delta = self.valid_delta(img[:,None,...], delta)
        #delta = torch.clamp(5 * delta + self.stripes[run_idx, ...], min=-self.eps, max=self.eps)
        #delta += self.stripes[run_idx, ...]
        inp = delta + img[:,None,...]
        inp = self.ensure_img_range(inp, self.img_range)
        return inp.flatten(end_dim=1)

    def sample_latent(self, dyn):
        out = self.rect_noise(dyn)
        s = dyn.drift.shape
        if self.antithetic:
            out[:, :s[1]//2, ...] = -out[:, -s[1]//2:, ...]
        return out