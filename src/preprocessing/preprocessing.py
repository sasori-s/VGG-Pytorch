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
from core.settings import Settings

settings = Settings()

class Preprocess(datasets.ImageFolder):
    def __init__(
            self, 
            dataset_mean : torch.tensor,
            dataset_std : torch.tensor,
            data_path : str, 
            scale_size : int = 256,
            purpose = 'single_scale_training',
            debug = True
    ) -> None: 
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std
        self.data_path = data_path
        self.image_size = settings.IMAGE_SIZE
        self.scale_size = scale_size
        self.batch_size = settings.BATCH_SIZE
        self.purpose = purpose
        self.debug = debug
        
    
    def find_classes(self, directory):
        if self.debug == False:
            return super().find_classes(directory)
            
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir()) 
        classes = classes[:len(classes * 1) // 6]
        
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        
        class_to_idx = {cls : i for i, cls in enumerate(classes)}
        return classes, class_to_idx
    
        
    def decide_transformation(self):
        if self.purpose == 'single_scale_training':
            super(Preprocess, self).__init__(self.data_path, transform=self.single_scale_training_transformations)
        elif self.purpose == 'mean_std_calculation':
            super(Preprocess, self).__init__(self.data_path, transform=self.mean_std_transformations)
        elif self.purpose == 'multiscale_training':
            super(Preprocess, self).__init__(self.data_path, transform=self.training_transformations)


    def all_transformations(self) -> None:
        self.training_transformations = v2.Compose([
            v2.Lambda(
                lambda image : v2.Resize(random.choice(settings.MULTISCLASS))(image)
            ),
            v2.RandomCrop(size=self.image_size),
            v2.RandomHorizontalFlip(),
            # v2.ToTensor(),# I believe this is redundant, as it converts PIL image and ndarray to Tensor 
            v2.ToImage(), 
            v2.ToDtype(dtype=torch.float32),
            v2.Normalize(mean=self.dataset_mean, std=self.dataset_std)
        ])

        self.single_scale_training_transformations = v2.Compose([
            v2.Resize(size=self.scale_size),
            v2.RandomCrop(size=self.image_size),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32),
            v2.Normalize(mean=self.dataset_mean, std=self.dataset_std),
        ])

        self.mean_std_transformations = v2.Compose([
            v2.Resize((self.image_size, self.image_size)),
            v2.ToTensor()
        ])


    def creating_datasets_and_dataloader(self):
        if self.purpose == 'single_scale_training':
            self.dataloader = DataLoader(self, batch_size=self.batch_size, shuffle=True, num_workers=3, pin_memory=True)
        else:
            self.dataloader = DataLoader(
                self, 
                batch_sampler=BatchSampler(batch_size=self.batch_size, drop_last=False),
                num_workers=8,
                pin_memory=True
            )
        
    
    def __call__(self):
        self.all_transformations()
        self.decide_transformation()
        self.creating_datasets_and_dataloader()
        return self.dataloader
    
    
if __name__ == '__main__':
    preproces = Preprocess("/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/CIFAR-100-dataset/train", 256)