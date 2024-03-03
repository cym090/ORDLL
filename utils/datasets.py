import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, KMNIST

class BaseData():
    def __init__(self, dir, transform=None, ):
        self.dir = dir
        self.transform = transform

class CIFAR10(BaseData):
    def __init__(self, dir, transform=None, **options):
        
        self.num_classes = 10
        self.dir = dir
        self.transform = transform
        # transform_train = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        # ])
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        # ])

        # batch_size = options['batch_size']
        # data_root = os.path.join(options['dataroot'], 'cifar10')

        # pin_memory = True if options['use_gpu'] else False

        trainset = torchvision.datasets.CIFAR10(root=self.dir, train=True, download=True, transform=self.transform)
        
        # trainloader = torch.utils.data.DataLoader(
        #     trainset, batch_size=batch_size, shuffle=True,
        #     num_workers=options['workers'], pin_memory=pin_memory,
        # )
        
        testset = torchvision.datasets.CIFAR10(root=self.dir, train=False, download=True, transform=self.transform)
        
        # testloader = torch.utils.data.DataLoader(
        #     testset, batch_size=batch_size, shuffle=False,
        #     num_workers=options['workers'], pin_memory=pin_memory,
        # )

        # self.trainloader = trainloader
        # self.testloader = testloader