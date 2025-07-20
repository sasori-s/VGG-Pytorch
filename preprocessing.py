import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
import os
from typing import List, Dict, Tuple
from torchvision.io import read_image

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
        self.all_transformations()
        self.decide_transformation(purpose)
        
        
    def decide_transformation(self, purpose):
        if purpose == 'training':
            super(Preprocess, self).__init__(self.data_path, transform=self.training_transformations)
        elif purpose == 'mean_std_calculation':
            super(Preprocess, self).__init__(self.data_path, transform=self.mean_std_transformations)


    def all_transformations(self) -> None:
        self.training_transformations = v2.Compose([
            v2.ScaleJitter(target_size=(self.scale_size, self.scale_size), scale_range=(0.1, 2.0)),
            v2.RandomCrop(size=self.image_size),
            v2.RandomHorizontalFlip(),
            v2.ToDtype(dtype=torch.float32),
            v2.Normalize(mean=[0.5087, 0.5006, 0.4405], std=[0.2832, 0.2682, 0.2887])
        ])

        self.single_scale_training_transformations = v2.Compose([
            v2.RandomCrop(size=self.image_size),
            v2.RandomHorizontalFlip(),
            v2.ToDtype(dtype=torch.float32),
            v2.Normalize(mean=[0.5087, 0.5006, 0.4405], std=[0.2832, 0.2682, 0.2887])
        ])

        self.mean_std_transformations = v2.Compose([
            v2.Resize((self.image_size, self.image_size)),
            v2.ToTensor()
        ])


    def creating_datasets_and_dataloader(self):
        self.dataloader = DataLoader(self, batch_size=128, shuffle=True, num_workers=3, pin_memory=True)
        
    
    


if __name__ == '__main__':
    preproces = Preprocess("/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/CIFAR-100-dataset/train", 256)