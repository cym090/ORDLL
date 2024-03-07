import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN
from omegaconf import ListConfig
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset


class CIFAR10_Filter(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __Filter__(self, class_list):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], [] #mask筛选的数据索引， new_targets重新命名类别
        for i in range(len(targets)):
            if targets[i] in class_list:
                mask.append(i)
                new_targets.append(class_list.index(targets[i]))
        self.data, self.targets = np.squeeze(np.take(datas, mask, axis=0)), np.array(new_targets)

class CIFAR10_OSR(object):
    def __init__(self, class_list, data_dir='./data/cifar10', train=True, transform=None,):
    # def __init__(self, class_list, dataroot='./data/cifar10', train=True, transform=None, use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.dir = data_dir
        self.class_num_ = len(class_list)
        self.class_list = class_list
        self.train = train
        # self.unclass_list = list(set(list(range(0, 10))) - set(class_list))
        if isinstance(transform, (list, ListConfig)):
            transform = transforms.Compose(transform)
        self.transform = transform
        # print('Selected Labels: ', class_list)

        # train_transform = transforms.Compose([
        #     transforms.Resize((img_size, img_size)),
        #     transforms.RandomCrop(img_size, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        # ])

        # transform = transforms.Compose([
        #     transforms.Resize((img_size, img_size)),
        #     transforms.ToTensor(),
        # ])

        # pin_memory = True if use_gpu else False

        self.data = CIFAR10_Filter(root=self.dir, train=self.train, download=True, transform=self.transform)
        # print('All Train Data:', len(data))
        self.data.__Filter__(class_list=self.class_list)
        
        # self.train_loader = torch.utils.data.DataLoader(
        #     data, batch_size=batch_size, shuffle=True,
        #     num_workers=num_workers, pin_memory=pin_memory,
        # )
        
        # testset = CIFAR10_Filter(root=self.dir, train=False, download=True, transform=transform)
        # print('All Test Data:', len(testset))
        # testset.__Filter__(class_list=self.class_list)
        
        # self.test_loader = torch.utils.data.DataLoader(
        #     testset, batch_size=batch_size, shuffle=False,
        #     num_workers=num_workers, pin_memory=pin_memory,
        # )

        # outset = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)
        # outset.__Filter__(class_list=self.unclass_list)

        # self.out_loader = torch.utils.data.DataLoader(
        #     outset, batch_size=batch_size, shuffle=False,
        #     num_workers=num_workers, pin_memory=pin_memory,
        # )

        # print('Train: ', len(data), 'Test: ', len(testset), 'Out: ', len(outset))
        # print('All Test: ', (len(testset) + len(outset)))
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

class CustomDataset(Dataset):#需要继承data.Dataset
    # def __init__(self, class_list, data_dir, train=True, transform=None):
    def __init__(self, data_dir, mode="train", transform=None):
        if isinstance(transform, (list, ListConfig)):
            transform = transforms.Compose(transform)
        self.transform = transform
        # self.dir = file_path
        # data, targets = self.make_dataset_from_folder(file_path)
        # self.data, self.targets = data, targets
        self.dataset = ImageFolder(f"{data_dir}/{mode}", transform=transform)
        # self.class_num_ = len(class_list)
        # self.class_list = class_list
        # self.__Filter__(class_list)

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
    
    def __Filter__(self, class_list):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in class_list:
                mask.append(i)
                new_targets.append(class_list.index(targets[i]))
        self.data, self.targets = np.squeeze(np.take(datas, mask, axis=0)), np.array(new_targets)
    
    def make_dataset_from_folder(file_path):
        dataset = ImageFolder(file_path)
        return dataset.data, dataset.targets