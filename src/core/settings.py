from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import torch

class Settings(BaseSettings):
    MODEL_NAME : str = 'VGG'
    DEBUG : bool = False
    
    DEVICE : str = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATASET_PATH1 : str
    DATASET_PATH2 : str
    
    IMAGE_SIZE : int = 224
    BATCH_SIZE : int = 64
    
    DATASET_MEAN_PATH : str = 'output/tensors/dataset_mean.pt'
    DATASET_STD_PATH : str = 'output/tensors/dataset_std.pt'
    DATASET_MEAN : torch.tensor = torch.tensor([0.4531, 0.4512, 0.3915])
    DATASET_STD : torch.tensor = torch.tensor([0.2573, 0.2421, 0.2585])
    
    MULTISCLASS : list[int] = [256, 384, 512]
    
    model_config = SettingsConfigDict(env_file='.env', extra='allow')