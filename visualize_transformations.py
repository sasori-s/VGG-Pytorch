from preprocessing import Preprocess
import matplotlib.pyplot as plt
import PIL
import numpy as np
from typing_extensions import Union, Tuple, List, Dict
import os
from pathlib import Path
from torchvision.datasets.folder import make_dataset
import random

class ShowImage(Preprocess):
    def __init__(self, data_path, scale_size, purpose='training'):
        super(ShowImage, self).__init__(data_path, scale_size, purpose)

    def find_classes(self, directory):
        k = 10
        classes, class_to_idx =  super().find_classes(directory)
        classes = classes[:k]
        class_to_idx = {key: value for key, value in list(class_to_idx.items())[:k]}
        
        return classes, class_to_idx


    def take_random_samples(self):
        self.random_image_indices = random.sample(range(len(self.imgs)), 10)
        self.random_images_tensor = [self[i] for i in self.random_image_indices]
        print(self.random_images_tensor)
    
    def show_image(self):
        figsize = [6, 8]
        nrow = 2
        ncol = 5

        fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=figsize)

        for i, ax_i in enumerate(ax.flat):
            image = self.random_images_tensor[i]
            ax_i.imshow(image)

            rowid = i // ncol
            colid = i % ncol

            ax_i.set_title("Add the class name of the sample") #Remember to add this

        
        plt.show()

        
    
    def __call__(self):
        self.take_random_samples()

if __name__ == '__main__':
    data_path = "/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/CIFAR-100-dataset/train"
    graph = ShowImage(data_path, 256)
    graph()