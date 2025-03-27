import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.transforms import v2
import os


class Preprocess(datasets.ImageFolder):
    def __init__(
            self, 
            data_path : str, 
            image_size : int
    ) -> None: 
        # super(Preprocess, self).__init__(data_path)
        self.data_path = data_path
        self.image_size = image_size
        self.show_images()

    
    def load_reshaped_image(
            self, image_path : str
    ) -> Image:
        image = Image.open(image_path)
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        plt.axis('off')
        plt.imshow(image)
        plt.savefig("reshaped_image.png")
        plt.show()
        return image

    def show_images(self):
        images = os.listdir(self.data_path)
        single_image_path = os.path.join(self.data_path, images[4])
        self.load_reshaped_image(single_image_path)
        print(len(os.listdir(self.data_path)))

    
    def transformations(self) -> None:
        transforms = v2.Compose([
            v2.RandomResizedCrop(self.image_size, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True)
        ])


if __name__ == '__main__':
    preproces = Preprocess("/teamspace/s3_connections/computer-vision-example/train/Angora", 256)