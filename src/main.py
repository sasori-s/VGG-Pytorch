from training.model import VGG11
from training.train import Training
from preprocessing.preprocessing import Preprocess
from preprocessing.precompute_mean_std import PreCompute
import torch.nn as nn
import torch
import os
from torch.optim.lr_scheduler import LRScheduler, StepLR
from dotenv import load_dotenv
from core.settings import Settings
from core.logger import logger

load_dotenv()

settings = Settings()

DATASET_PATH = settings.DATASET_PATH1 if os.path.exists(settings.DATASET_PATH1) else settings.DATASET_PATH2
DEVICE = settings.DEVICE

settings.DEBUG = True if DATASET_PATH == settings.DATASET_PATH1 else False


def compute_mean_std_of_dataset():
    logger.warning(f"Debug mode --- {settings.DEBUG}")
    logger.info(f"Computing mean and standarnd deviation for the dataset")
    
    if settings.DEBUG is False:
        if os.path.exists(settings.DATASET_MEAN_PATH) and os.path.exists(settings.DATASET_STD_PATH):
            mean = torch.load(settings.DATASET_MEAN_PATH)
            std = torch.load(settings.DATASET_STD_PATH)
        else:
            precompute_mean_std = PreCompute(
                data_path=os.path.join(DATASET_PATH, 'train'),
                debug=settings.DEBUG
            )
            
            mean, std = precompute_mean_std()
    else:
        mean = settings.DATASET_MEAN
        std = settings.DATASET_STD
        
    return mean, std


def initiate_training_parameters():
    mean, std = compute_mean_std_of_dataset()
    
    train_preprocessor = Preprocess(
        os.path.join(DATASET_PATH, 'train'), 
        scale_size=256, 
        purpose='single_scale_training'
    )
    
    val_preprocessor = Preprocess(os.path.join(DATASET_PATH, 'val'))

    train_dataloader = train_preprocessor()
    val_dataloader = val_preprocessor()

    criterion = nn.CrossEntropyLoss()

    model = VGG11(torch.randn(32, 3, 224, 224).to(DEVICE), num_classes=100)
    optimizer_sgd = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=5 * 10e-4, momentum=0.9)
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-2)
    lr_scheduler = StepLR(optimizer_sgd, step_size=15, gamma=0.01)
    epochs = 50
    batch_size = 64

    model_kwgs = {
        "model" : model,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_preprocessor,
        "epochs" : epochs,
        "criterion" : criterion,
        "optimizer" : optimizer_sgd,
        "lr_scheduler" : lr_scheduler,
        "batch_size" : batch_size,

    } 

    return model_kwgs
    

def main():
    kwags = initiate_training_parameters()
    trainer = Training(**kwags)
    

if __name__ == '__main__':
    main()
    


