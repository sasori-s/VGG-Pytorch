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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Preprocess(datasets.ImageFolder):
    def __init__(
            self, 
            data_path : str, 
            image_size : int,
            single_image : bool = False
    ) -> None: 
        self.data_path = data_path
        self.image_size = image_size
        self.transformations()
        # self.show_images()
        super(Preprocess, self).__init__(data_path, transform=self.transforms)
        self.dataloader = DataLoader(self, batch_size=32, shuffle=True)
        print(self.classes)


    def load_reshaped_image(
            self, image_path : str
    ) -> Image:
        image = Image.open(image_path)
        # image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        # image = v2.Resize(size = (self.image_size, self.image_size), antialias=True)(image)
        plt.axis('off')
        plt.imshow(image)
        plt.savefig("reshaped_image.png")
        plt.show()

        transformed_image = self.transforms(image)
        tensor_to_image = transformed_image.numpy().transpose(1, 2, 0)
        plt.imshow(tensor_to_image)
        plt.savefig("transformed_image.png")

        tensor_image = v2.ToImage()(image)
        tensor_to_image = tensor_image.numpy().transpose(1, 2, 0)
        plt.imshow(tensor_to_image)
        plt.savefig("tensor_img.png")
        return image
    

    def show_images(self) -> None:
        images = os.listdir(self.data_path)
        single_image_path = os.path.join(self.data_path, images[1])
        self.load_reshaped_image(single_image_path)
        print(len(os.listdir(self.data_path)))

    
    def transformations(self) -> None:
        self.transforms = v2.Compose([
            v2.Resize(size=(self.image_size, self.image_size), antialias=True),
            v2.ToImage(),
            v2.RandomResizedCrop(self.image_size, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True)
        ])


    def calculate_mean_std_for_dataset(self) -> Tuple[List[int], List[int]]:
        channel_sum = torch.zeros(3)
        channel_squared_sum = torch.zeros(3)
        num_pixels = 0

        for images, _ in self.dataloader:
            images = images.to(device)
            num_pixels += images.size(0)
            channel_sum += torch.sum(images, dim=(0, 2, 3))
            channel_squared_sum += torch.sun(images ** 2, dim=(0, 2, 3))
        
        mean = channel_sum / num_pixels
        std = torch.sqrt(channel_squared_sum / num_pixels - mean ** 2)
        return mean.tolist(), std.tolist()


if __name__ == '__main__':
    preproces = Preprocess("/teamspace/s3_connections/computer-vision-example/train", 256)