import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
import os
import random
from typing import List, Dict, Tuple
from torch.utils.data.sampler import BatchSampler

MULTISCLASS = [256, 384, 512]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Preprocess(datasets.ImageFolder):
    def __init__(
            self, 
            data_path : str, 
            scale_size : int = 256,
            purpose = 'training'
    ) -> None: 
        
        self.data_path = data_path
        self.image_size = 224
        self.scale_size = scale_size
        self.batch_size = 64
        self.purpose = purpose
        
        
    def decide_transformation(self):
        if self.purpose == 'training':
            super(Preprocess, self).__init__(self.data_path, transform=self.single_scale_training_transformations)
        elif self.purpose == 'mean_std_calculation':
            super(Preprocess, self).__init__(self.data_path, transform=self.mean_std_transformations)
        elif self.purpose == 'multiscale_training':
            super(Preprocess, self).__init__(self.data_path, transform=self.training_transformations)


    def all_transformations(self) -> None:
        self.training_transformations = v2.Compose([
            v2.Lambda(
                lambda image : v2.Resize(random.choice(MULTISCLASS))(image)
            ),
            v2.RandomCrop(size=self.image_size),
            v2.RandomHorizontalFlip(),
            v2.ToDtype(dtype=torch.float32),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2623, 0.2513, 0.2714])
        ])

        self.single_scale_training_transformations = v2.Compose([
            v2.Resize(size=self.scale_size),
            v2.RandomCrop(size=self.image_size),
            v2.RandomHorizontalFlip(),
            v2.ToDtype(dtype=torch.float32),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2623, 0.2513, 0.2714]),
        ])

        self.mean_std_transformations = v2.Compose([
            v2.Resize((self.image_size, self.image_size)),
            v2.ToTensor()
        ])


    def creating_datasets_and_dataloader(self):
        self.dataloader = DataLoader(self, batch_size=128, shuffle=True, num_workers=3, pin_memory=True)
        self.multiscale_dataloader = DataLoader(
            self, 
            batch_sampler=BatchSampler(batch_size=self.batch_size, drop_last=False),
            num_workers=8,
            pin_memory=True
        )
        
    
    def __call__(self):
        self.all_transformations()
        self.decide_transformation(self.purpose)
        return self.dataloader
    
    
if __name__ == '__main__':
    preproces = Preprocess("/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/CIFAR-100-dataset/train", 256)