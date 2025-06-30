import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import os
from .CNN import CNN
from .cifar_resnet import ResNet50
from collections import OrderedDict

class PreProcessedModel(nn.Module):
    def __init__(self, model, transform):
        super().__init__()
        self.main_model = model
        self.transform = transform

    def forward(self, x):
        x = self.transform(x)
        return self.main_model(x)

def get_preprocessed_model_ImageNet(model):
    trafo = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
    return PreProcessedModel(model, trafo)

def get_preprocessed_model_CIFAR(model):
    trafo = transforms.Compose([
                transforms.Normalize(mean = (0.4914, 0.4822, 0.4465), 
                                     std = (0.2023, 0.1994, 0.2010))
            ])
    return PreProcessedModel(model, trafo)


def load_CNN(cfg):
    model = CNN(mean = cfg.data.mean, std = cfg.data.std, 
                       ksize1 = 5, ksize2 = 5, stride = 1).to(cfg.device)
    model.load_state_dict(torch.load(cfg.model.path, map_location=cfg.device)['net'])
    model.eval()
    return model

def load_MadryMNIST(cfg):
    from .madry_mnist import ModelTF
    return ModelTF('madry_mnist_robust', cfg.data.batch_size_test, .5)
    




def load_CIFAR_ResNet(cfg):
    model = ResNet50()
    state_dict = OrderedDict()
    for key, v in torch.load(cfg.model.path, map_location=cfg.device)['net'].items():
        keys = key.split('.')
        if keys[0] == 'module':
            keys = keys[1:]
        key = '.'.join(keys)
        state_dict[key] = v

    model.load_state_dict(state_dict)
    model.eval()
    return get_preprocessed_model_CIFAR(model)

def load_ConvNext(cfg):
    model = torchvision.models.convnext_small(weights = 'DEFAULT')
    model.eval()
    return get_preprocessed_model_ImageNet(model)

def load_ResNet(cfg):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    model.eval()
    return get_preprocessed_model_ImageNet(model)

def load_AlexNet(cfg):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    model.eval()
    # update imsize to fir the description here:
    # https://arxiv.org/pdf/1710.08864
    cfg.data.shape = [3, 227, 227] 
    cfg.data.resize_width = 227
    return get_preprocessed_model_ImageNet(model)

def load_DenseNet(cfg):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
    model.eval()
    return get_preprocessed_model_ImageNet(model)

def load_VGG(cfg):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=True)
    model.eval()
    return get_preprocessed_model_ImageNet(model)

def load_inception_v3(cfg):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    model.eval()
    cfg.data.shape = [3, 299, 299] # update imsize
    cfg.data.resize_width = 342
    return get_preprocessed_model_ImageNet(model)

model_dict = {'CNN': load_CNN, 'ConvNext':load_ConvNext, 
              'Inception_v3':load_inception_v3, 
              'DenseNet': load_DenseNet, 
              'ResNet': load_ResNet, 
              'ResNetCIFAR': load_CIFAR_ResNet, 
              'AlexNet': load_AlexNet, 
              'VGG': load_VGG,
              'MadryMNIST': load_MadryMNIST}

def load_model(cfg):
    name = cfg.model.name
    if name in model_dict.keys():
        return model_dict[name](cfg)
    else:
        raise ValueError('Unknown model: ' +str(name))



    