import torch

def init_stripes(x, f=0.05):
    return torch.zeros_like(x) + f * (1-2 * torch.bernoulli(0.5 * torch.ones_like(x[..., :1, :])))

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
    def __init__(self, max_calls=1e5, p_init=0.05, im_w = 299, num_rects = 1):
        self.max_calls = max_calls
        self.p_init = p_init
        self.num_calls = 0
        self.im_w = im_w
        self.num_rects = num_rects
    
    def __call__(self, dyn):
        self.select_w(dyn.drift.shape[1], dyn.batch_size)
        out = torch.zeros_like(dyn.drift)
        for i in range(self.num_rects):
            out += rectangle_sampler(dyn.drift, w=self.w//(i+1), fix_channel_pos = True)
        return out


    def select_w(self, k, bs=6):
        self.num_calls += bs//2
        self.w = int(p_selection(self.p_init, self.num_calls, self.max_calls)**0.5 * self.im_w)