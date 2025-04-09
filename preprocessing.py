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
        self._calculate_mean_std_vectorized()
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
            channel_squared_sum += torch.sum(images ** 2, dim=(0, 2, 3))
        
        mean = channel_sum / num_pixels
        std = torch.sqrt(channel_squared_sum / num_pixels - mean ** 2)
        return mean.tolist(), std.tolist()


    def _calculate_mean_std(self):
        means = []
        variances = []
        images_rgb = [np.array(Image.open(images[0]).getdata()) / 255  for images in self.imgs[:100]]

        for image in images_rgb:
            if len(image.shape) == 2:
                means.append(np.mean(image, axis=0))

        mean = np.mean(means, axis=0)

        for image in images_rgb:
            if len(image.shape) == 2:
                var = np.mean((image - mean) ** 2, axis=0)
                variances.append(var)
        
        std = np.sqrt(np.mean(variances, axis=0))
        return mean, std

    
    def _calculate_mean_std_vectorized(self):
        
        def check(image):
            image = np.array(Image.open(image).getdata()) / 255
            if len(image.shape) == 2:
                return True
            
            return False

        images_rgb = np.concatenate(
            [Image.open(image[0]).getdata() if check(image[0]) else np.zeros((256 * 256, 3)) for image in self.imgs[:100]],
            axis=0
        ) / 255

        mean = np.mean(images_rgb, axis=0)
        std = np.std(images_rgb, axis=0)
            
        return mean, std

    


if __name__ == '__main__':
    preproces = Preprocess("/teamspace/s3_connections/computer-vision-example/train", 256)