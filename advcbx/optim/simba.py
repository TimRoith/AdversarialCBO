import torch
import torch.nn.functional as F
import torchvision.transforms as trans
#import utils
import numpy as np
from scipy.fftpack import dct, idct

# mean and std for different datasets
IMAGENET_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_TRANSFORM = trans.Compose([
    trans.Resize(256),
    trans.CenterCrop(224),
    trans.ToTensor()])

INCEPTION_SIZE = 299
INCEPTION_TRANSFORM = trans.Compose([
    trans.Resize(342),
    trans.CenterCrop(299),
    trans.ToTensor()])

CIFAR_SIZE = 32
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]
CIFAR_TRANSFORM = trans.Compose([
    trans.ToTensor()])

MNIST_SIZE = 28
MNIST_MEAN = [0.5]
MNIST_STD = [1.0]
MNIST_TRANSFORM = trans.Compose([
    trans.ToTensor()])

def block_order(image_size, channels, initial_size=1, stride=1):
    order = torch.zeros(channels, image_size, image_size)
    total_elems = channels * initial_size * initial_size
    perm = torch.randperm(total_elems)
    order[:, :initial_size, :initial_size] = perm.view(channels, initial_size, initial_size)
    for i in range(initial_size, image_size, stride):
        num_elems = channels * (2 * stride * i + stride * stride)
        perm = torch.randperm(num_elems) + total_elems
        num_first = channels * stride * (stride + i)
        order[:, :(i+stride), i:(i+stride)] = perm[:num_first].view(channels, -1, stride)
        order[:, i:(i+stride), :i] = perm[num_first:].view(channels, stride, -1)
        total_elems += num_elems
    return order.view(1, -1).squeeze().long().sort()[1]


# applies IDCT to each block of size block_size
def block_idct(x, block_size=8, masked=False, ratio=0.5, linf_bound=0.0):
    z = torch.zeros(x.size())
    num_blocks = int(x.size(2) / block_size)
    mask = np.zeros((x.size(0), x.size(1), block_size, block_size))
    if type(ratio) != float:
        for i in range(x.size(0)):
            mask[i, :, :int(block_size * ratio[i]), :int(block_size * ratio[i])] = 1
    else:
        mask[:, :, :int(block_size * ratio), :int(block_size * ratio)] = 1
    for i in range(num_blocks):
        for j in range(num_blocks):
            submat = x[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)].numpy()
            if masked:
                submat = submat * mask
            z[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)] = torch.from_numpy(idct(idct(submat, axis=3, norm='ortho'), axis=2, norm='ortho'))
    if linf_bound > 0:
        return z.clamp(-linf_bound, linf_bound)
    else:
        return z

# applies the normalization transformations
def apply_normalization(imgs, dataset):
    if dataset == 'imagenet':
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    elif dataset == 'cifar':
        mean = CIFAR_MEAN
        std = CIFAR_STD
    elif dataset == 'mnist':
        mean = MNIST_MEAN
        std = MNIST_STD
    else:
        mean = [0, 0, 0]
        std = [1, 1, 1]
    imgs_tensor = imgs.clone()
    if dataset == 'mnist':
        imgs_tensor = (imgs_tensor - mean[0]) / std[0]
    else:
        if imgs.dim() == 3:
            for i in range(imgs_tensor.size(0)):
                imgs_tensor[i, :, :] = (imgs_tensor[i, :, :] - mean[i]) / std[i]
        else:
            for i in range(imgs_tensor.size(1)):
                imgs_tensor[:, i, :, :] = (imgs_tensor[:, i, :, :] - mean[i]) / std[i]
    return imgs_tensor


class SimBA:
    def __init__(
        self, model, space, 
        x_orig, y, 
        max_evals=10000,
        eps = 0.2,
        targeted=False,
        dataset='imagenet',
        max_iters = 10000,
        freq_dims = 38,
        stride    = 9,
        order     = 'strided',
        pixel_attack = False,
        log_every    = 10,
        linf_bound   = 0.
    ):
        self.model = model
        self.image_size = x_orig.shape[-1]
        self.x_orig       = x_orig
        self.y            = y
        self.max_evals    = max_evals
        self.epsilon      = eps
        self.targeted     = targeted
        self.dataset      = dataset

        self.max_iters    = max_iters
        self.freq_dims    = freq_dims  
        self.stride       = stride
        self.order        = order 
        self.pixel_attack = pixel_attack
        self.log_every    = log_every
        self.linf_bound   = linf_bound
    
    def expand_vector(self, x, size):
        batch_size = x.size(0)
        x = x.view(-1, 3, size, size)
        z = torch.zeros(batch_size, 3, self.image_size, self.image_size)
        z[:, :, :size, :size] = x
        return z
        
    def normalize(self, x):
        return x
        #return apply_normalization(x, self.dataset)
    
    def get_best_img(self,):
        return self.x_best #self.normalize(self.x.cuda())

    def get_num_queries(self,):
        return self.queries.cpu().numpy().sum(-1)

    def get_cur_energy(self,):
        return self.probs

    def get_probs(self, x, y):
        self.num_evals += x.shape[0]
        output = self.model(self.normalize(x.cuda()))#.cpu()
        probs = torch.index_select(F.softmax(output, dim=-1).data, 1, y)
        return torch.diag(probs)
    
    def get_preds(self, x):
        self.num_evals += x.shape[0]
        output = self.model(self.normalize(x.cuda()))#.cpu()
        _, preds = output.data.max(1)
        return preds

    # # 20-line implementation of SimBA for single image input
    # def optimize(self,):
    #     self.num_evals = 0
    #     n_dims = self.x.view(1, -1).size(1)
    #     perm = torch.randperm(n_dims)
    #     #self.x = self.x.unsqueeze(0)
    #     last_prob = self.get_probs(self.x, self.y)
    #     i = 0
    #     while self.num_evals < self.max_evals:
    #         diff = torch.zeros(n_dims).to(self.x.device)
    #         diff[perm[i]] = self.epsilon
    #         left_prob = self.get_probs((self.x - diff.view(self.x.size())).clamp(0, 1), self.y)
    #         if self.targeted != (left_prob < last_prob):
    #             self.x = (self.x - diff.view(self.x.size())).clamp(0, 1)
    #             last_prob = left_prob
    #         else:
    #             right_prob = self.get_probs((self.x + diff.view(self.x.size())).clamp(0, 1), self.y)
    #             if self.targeted != (right_prob < last_prob):
    #                 self.x = (self.x + diff.view(self.x.size())).clamp(0, 1)
    #                 last_prob = right_prob
    #         if i % 100 == 0:
    #             print('Num Queries: ' + str(self.num_evals))
    #             print(last_prob)
    #         i+=1
    #     return self.x.squeeze()

    # runs simba on a batch of images <images_batch> with true labels (for untargeted attack) or target labels
    # (for targeted attack) <labels_batch>
    def optimize(self, ):
        freq_dims, max_iters, stride, order = self.freq_dims, self.max_iters, self.stride, self.order
        images_batch, labels_batch = self.x_orig, self.y
        self.num_evals = 0
        batch_size = images_batch.shape[0]
        image_size = images_batch.shape[2]
        #assert self.image_size == image_size
        # sample a random ordering for coordinates independently per batch element
        if order == 'rand':
            indices = torch.randperm(3 * freq_dims * freq_dims)[:max_iters]
        elif order == 'diag':
            indices = utils.diagonal_order(image_size, 3)[:max_iters]
        elif order == 'strided':
            indices = block_order(image_size, 3, initial_size=freq_dims, stride=stride)[:max_iters]
        else:
            indices = utils.block_order(image_size, 3)[:max_iters]
        if order == 'rand':
            expand_dims = freq_dims
        else:
            expand_dims = image_size
        n_dims = 3 * expand_dims * expand_dims
        x = torch.zeros(batch_size, n_dims)
        # logging tensors
        probs = torch.zeros(batch_size, max_iters)
        succs = torch.zeros(batch_size, max_iters)
        queries = torch.zeros(batch_size, max_iters)
        l2_norms = torch.zeros(batch_size, max_iters)
        linf_norms = torch.zeros(batch_size, max_iters)
        prev_probs = self.get_probs(images_batch, labels_batch).cpu()
        preds = self.get_preds(images_batch)
        if self.pixel_attack:
            trans = lambda z: z
        else:
            trans = lambda z: block_idct(z, block_size=image_size, linf_bound= self.linf_bound).to(images_batch.device)
        remaining_indices = torch.arange(0, batch_size).long()
        for k in range(max_iters):
            dim = indices[k]
            expanded = (images_batch[remaining_indices] + trans(self.expand_vector(x[remaining_indices], expand_dims))).clamp(0, 1)
            perturbation = trans(self.expand_vector(x, expand_dims))
            l2_norms[:, k] = perturbation.view(batch_size, -1).norm(2, 1)
            linf_norms[:, k] = perturbation.view(batch_size, -1).abs().max(1)[0]
            preds_next = self.get_preds(expanded)
            preds[remaining_indices] = preds_next
            if self.targeted:
                remaining = preds.ne(labels_batch).cpu()
            else:
                remaining = preds.eq(labels_batch).cpu()
            # check if all images are misclassified and stop early
            if remaining.sum() == 0:
                adv = (images_batch + trans(self.expand_vector(x, expand_dims))).clamp(0, 1)
                probs_k = self.get_probs(adv, labels_batch)
                probs[:, k:] = probs_k.unsqueeze(1).repeat(1, max_iters - k)
                succs[:, k:] = torch.ones(batch_size, max_iters - k)
                queries[:, k:] = torch.zeros(batch_size, max_iters - k)
                break
            remaining_indices = torch.arange(0, batch_size)[remaining].long()
            if k > 0:
                succs[:, k] = ~remaining
            diff = torch.zeros(remaining.sum(), n_dims)
            diff[:, dim] = self.epsilon
            left_vec     = x[remaining_indices] - diff
            right_vec    = x[remaining_indices] + diff
            # trying negative direction
            adv = (images_batch[remaining_indices] + trans(self.expand_vector(left_vec, expand_dims))).clamp(0, 1)
            left_probs = self.get_probs(adv, labels_batch[remaining_indices]).cpu()
            queries_k = torch.zeros(batch_size)
            # increase query count for all images
            queries_k[remaining_indices] += 1
            if self.targeted:
                improved = left_probs.gt(prev_probs[remaining_indices]).cpu()
            else:
                improved = left_probs.lt(prev_probs[remaining_indices]).cpu()
            # only increase query count further by 1 for images that did not improve in adversarial loss
            if improved.sum() < remaining_indices.size(0):
                queries_k[remaining_indices[~improved]] += 1
            # try positive directions
            adv = (images_batch[remaining_indices] + trans(self.expand_vector(right_vec, expand_dims))).clamp(0, 1)
            right_probs = self.get_probs(adv, labels_batch[remaining_indices]).cpu()
            if self.targeted:
                right_improved = right_probs.gt(torch.max(prev_probs[remaining_indices], left_probs)).cpu()
            else:
                right_improved = right_probs.lt(torch.min(prev_probs[remaining_indices], left_probs)).cpu()
            probs_k = prev_probs.clone()
            # update x depending on which direction improved
            if improved.sum() > 0:
                left_indices = remaining_indices[improved]
                left_mask_remaining = improved.unsqueeze(1).repeat(1, n_dims)
                x[left_indices] = left_vec[left_mask_remaining].view(-1, n_dims)
                probs_k[left_indices] = left_probs[improved]
            if right_improved.sum() > 0:
                right_indices = remaining_indices[right_improved]
                right_mask_remaining = right_improved.unsqueeze(1).repeat(1, n_dims)
                x[right_indices] = right_vec[right_mask_remaining].view(-1, n_dims)
                probs_k[right_indices] = right_probs[right_improved]
            probs[:, k] = probs_k
            queries[:, k] = queries_k
            prev_probs = probs[:, k]
            if (k + 1) % self.log_every == 0 or k == max_iters - 1:
                print('Iteration %d: queries = %.4f, prob = %.4f, remaining = %.4f' % (
                        k + 1, queries.sum(1).mean(), probs[:, k].mean(), remaining.float().mean()))
        expanded = (images_batch + trans(self.expand_vector(x, expand_dims))).clamp(0, 1)
        preds = self.get_preds(expanded)
        if self.targeted:
            remaining = preds.ne(labels_batch)
        else:
            remaining = preds.eq(labels_batch)
        succs[:, max_iters-1] = ~remaining
        self.x_best  = expanded
        self.queries = queries
        self.probs   = probs
        # return expanded, probs, succs, queries, l2_norms, linf_norms