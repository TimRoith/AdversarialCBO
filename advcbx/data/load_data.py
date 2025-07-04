from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import ImageFolder
import torch
import PIL
import numpy as np
import difflib

import os

known_datasets = ['MNIST', 'ImageNet']
#%% 
def load_data(cfg):
    if cfg.data.name == "MNIST":
        return load_MNIST_test(cfg)
    elif cfg.data.name == 'ImageNet':
        return load_ImageNet_test(cfg)
    elif cfg.data.name == 'CIFAR10':
        return load_CIFAR10_test(cfg)
    else:
        raise ValueError("Unknown dataset: " + cfg.data.name + '. You can chose from: ' + str(known_datasets))
        
class resize_to_range:
    def __init__(self, xmin=0.,xmax=1.):
        self.xmin=xmin
        self.xmax=xmax
    def __call__(self, img):
        ishape = img.shape
        img = img.view(*ishape[:-2], -1)
        #img -= img.min(-1, keepdim=True)[0]
        img /= img.max(-1, keepdim=True)[0]
        return img.view(ishape)

def load_MNIST_test(cfg):
    cfgd = cfg.data
    #cfgd.shape = [1, 28, 28]
    transform = transforms.Compose([transforms.ToTensor()])
    loader_kwargs = {'pin_memory':True, 'num_workers':cfgd.num_workers}
    test = datasets.MNIST(cfgd.path, train=False, download=cfgd.download, transform=transform)
    test_loader = DataLoader(test, batch_size=cfgd.batch_size_test, **loader_kwargs)
    return test_loader

# get center crop
def load_image(path):
    image = PIL.Image.open(path)
    if image.height > image.width:
        height_off = int((image.height - image.width)/2)
        image = image.crop((0, height_off, image.width, height_off+image.width))
    elif image.width > image.height:
        width_off = int((image.width - image.height)/2)
        image = image.crop((width_off, 0, width_off+image.height, image.height))
    image = image.resize((299, 299))
    img = np.asarray(image).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.repeat(img[:,:,np.newaxis], repeats=3, axis=2)
    if img.shape[2] == 4:
        # alpha channel
        img = img[:,:,:3]
    return img

def load_from_image_net(imagenet_path, idx, idx2=0):
    data_path = os.path.join(imagenet_path, 'val')
    image_paths = sorted([os.path.join(data_path, i) for i in os.listdir(data_path)])
    labels_path = os.path.join(imagenet_path, 'val.txt')
    with open(labels_path) as labels_file:
        labels = [i.split(' ') for i in labels_file.read().strip().split('\n')]
        labels = {os.path.basename(i[0]): j for j,i in enumerate(labels)}
    def get(index):
        path = image_paths[index]        
        x = load_image(os.path.join(path, sorted(os.listdir(path))[idx2]))
        y = labels[os.path.basename(path)]
        return x, y
    return get(idx)

def load_single_image_and_class(path, cfg, classes = None):
    img = PIL.Image.open(path)
    im_width = cfg.data.shape[-1]
    resize_width = getattr(cfg.data, 'resize_width', 256)
    transform = transforms.Compose([
        transforms.Resize(resize_width),
        transforms.CenterCrop(im_width),
        transforms.ToTensor(),
    ])
    img = transform(img)[None, :3, ...]
    img = (img - img.min())/(img.max() - img.min())

    if classes is not None:
        class_name = path.split('/')[-1].split('.')[0].split('-')[0]
        class_idx = next((i for i,v in enumerate(classes) if 
                  (len(difflib.get_close_matches(
                      class_name, 
                      (k for g in v for k in g.split(' ')), n=1, cutoff=0.65)
                      ) > 0)
                ), None)
    if class_idx is None: class_idx = 0
    return img.to(cfg.device), torch.tensor([class_idx], device = cfg.device)
    
def load_CIFAR10_test(cfg):
    cfgd = cfg.data
    im_width = cfg.data.shape[-1]
    resize_width = getattr(cfg.data, 'resize_width', 32)

    transform_test = transforms.Compose([
        transforms.Resize(resize_width),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    val = torchvision.datasets.CIFAR10(cfgd.path, train=False, download=True, transform=transform_test)
    loader_kwargs = {'pin_memory':True, 'num_workers':cfgd.num_workers, 'shuffle': cfgd.shuffle}
    val_loader = DataLoader(val, batch_size=cfgd.batch_size_test, **loader_kwargs)
    return val_loader


def load_ImageNet_test(cfg):
    cfgd = cfg.data
    im_width = cfg.data.shape[-1]
    resize_width = getattr(cfg.data, 'resize_width', 256)

    transform = transforms.Compose([
        transforms.Resize(resize_width),
        transforms.CenterCrop(im_width),
        transforms.ToTensor(),
    ])

    rng = torch.Generator()
    rng.manual_seed(cfg.seed + 8)
    
    val = torchvision.datasets.ImageNet(cfgd.path + '/ImageNet', split='val', transform=transform)
    sampler = RandomSampler(val, generator=rng)
    
    loader_kwargs = {'pin_memory':True, 'num_workers': 0, 'shuffle': False,
                     'sampler': sampler, 'persistent_workers':False}
    val_loader = DataLoader(val, batch_size=cfgd.batch_size_test, **loader_kwargs)
    return val_loader


#%% Define DataLoaders and split in train, valid, test       
def split_loader(train, test, valid=None, batch_size=128, batch_size_test=100,\
                 train_split=0.9, num_workers=0, seed=42):
    total_count = len(train)
    train_count = int(train_split * total_count)
    val_count = total_count - train_count
    generator=torch.Generator().manual_seed(seed)

    loader_kwargs = {'pin_memory':True, 'num_workers':num_workers}
    if not (valid is None):
        valid_loader = DataLoader(valid, batch_size=batch_size, **loader_kwargs)
    elif val_count > 0:
        train, valid = torch.utils.data.random_split(train,\
                                    [train_count, val_count],generator=generator)
        valid_loader = DataLoader(valid, batch_size=batch_size, **loader_kwargs)
    else:
        valid_loader = None
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test, batch_size=batch_size_test, **loader_kwargs)
    return train_loader, valid_loader, test_loader