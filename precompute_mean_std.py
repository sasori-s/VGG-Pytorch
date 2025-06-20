import torch
import numpy as np
from preprocessing import Preprocess
from torchvision.io import read_image
from torchvision.transforms import v2
from typing import List, Tuple
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PreCompute(Preprocess):
    def __init__(self, data_path, image_size):
        super(PreCompute, self).__init__(data_path, image_size)


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
    
    
    def calculate_mean_std_torch_vectorized(self):
        image_tensors = []

        def reshape_and_convert(image_path):
            image = Image.open(image_path).convert("RGB")
            image = v2.Resize(size=(self.image_size, self.image_size), antialias=True)(image)
            image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1)
            return image_tensor


        image_tensors = [
            reshape_and_convert(image[0]) for image in self.imgs[:]
        ]

        image_tensors = torch.stack(image_tensors).float() / 255.0
        image_tensors = image_tensors.to(device)
        mean = torch.mean(image_tensors, dim=(0, 2, 3)) #tensor([0.5087, 0.5006, 0.4405], device='cuda:0')
        std = torch.std(image_tensors, dim=(0, 2, 3)) #tensor([0.2832, 0.2682, 0.2887], device='cuda:0')
        return mean, std
    

    def __call__(self):
        mean, std = self.calculate_mean_std_torch_vectorized()
        print(mean, std)

    
if __name__ == '__main__':
    precompute = PreCompute("/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/animal_dataset/animals/animals", 256)
    precompute()