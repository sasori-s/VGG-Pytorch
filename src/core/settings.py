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
    
    model_config = SettingsConfigDict(env_file='.env', extra='allow')