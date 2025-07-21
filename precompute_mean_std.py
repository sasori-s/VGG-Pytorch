import torch
import numpy as np
from preprocessing import Preprocess
from torchvision.io import read_image
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from typing import List, Tuple
from PIL import Image
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PreCompute(Preprocess):
    def __init__(self, data_path, image_size):
        super(PreCompute, self).__init__(data_path, image_size, purpose='mean_std_calculation')
        self.dataloader = DataLoader(self, batch_size=256, num_workers=4, pin_memory=True)


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
    

    
    def compute_mean_std(self, batch_size=64, num_workers=4, device='cuda'):
        n_channels = 3
        mean = torch.zeros(n_channels).to(device)
        std = torch.zeros(n_channels).to(device)
        n_pixels = 0

        for images, _ in tqdm(self.dataloader, desc="computing mean/std"):
            images = images.to(device)
            b, c, w, h = images.shape
            n_pixels += b * w * h

            mean += images.sum(dim=[0, 2, 3])
            std += (images ** 2 ).sum(dim=[0, 2, 3])

        mean /= n_pixels #[tensor([0.5071, 0.4865, 0.4409], device='cuda:0')]
        std = torch.sqrt((std / n_pixels) - mean ** 2) #[tensor([0.2623, 0.2513, 0.2714], device='cuda:0')]

        return mean, std
    
    

    def __call__(self):
        mean, std = self.compute_mean_std()
        print(mean, std)

    
if __name__ == '__main__':
    precompute = PreCompute("/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/CIFAR-100-dataset/train", 256)
    precompute()