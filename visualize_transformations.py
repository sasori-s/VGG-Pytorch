from preprocessing import Preprocess
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from typing_extensions import Union, Tuple, List, Dict
import os
from pathlib import Path
from torchvision.datasets.folder import make_dataset
import random

class ShowImage(Preprocess):
    def __init__(self, data_path, scale_size, purpose='multiscale_training'):
        super(ShowImage, self).__init__(data_path, scale_size, purpose)

    def find_classes(self, directory):
        k = 10
        classes, class_to_idx =  super().find_classes(directory)
        classes = classes[:k]
        class_to_idx = {key: value for key, value in list(class_to_idx.items())[:k]}
        
        return classes, class_to_idx


    def take_random_samples(self):
        random_image_indices = random.sample(range(len(self.imgs)), 10)
        self.random_images_tensor = [self[i] for i in random_image_indices]

        self.idx_to_class = {value : key for key, value in self.class_to_idx.items()}

        self.non_transformed_images = [np.array(Image.open(self.imgs[i][0]).resize(size=(224, 224))) for i in random_image_indices]
        print(self.non_transformed_images)

    
    def show_image(self):
        figsize = [15, 6]
        nrow = 2
        ncol = 5

        fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=figsize)

        plt.suptitle("Top: Transformed Image | Bottom: Original Image")
        for i in range(5):
            transformed_image = self.random_images_tensor[i][0].permute(1, 2, 0).numpy()
            ax[0, i].imshow(transformed_image)
            ax[0, i].axis('off')
            ax[0, i].set_title(f"{self.idx_to_class[self.random_images_tensor[i][1]]}") #Remember to add this

            ax[1, i].imshow(self.non_transformed_images[i])
            ax[1, i].axis("off")
            ax[1, i].set_title(f"{self.idx_to_class[self.random_images_tensor[i][1]]}")

        plt.tight_layout()
        # plt.show()
        plt.savefig('original_vs_transformed.png')

        
    
    def __call__(self):
        self.take_random_samples()
        self.show_image()

if __name__ == '__main__':
    data_path = "/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/CIFAR-100-dataset/train"
    graph = ShowImage(data_path, 256)
    graph()