from model import VGG11
from train import Training
from preprocessing import Preprocess
from precompute_mean_std import PreCompute
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import LRScheduler, StepLR

DATASET_PATH = "/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/CIFAR-100-dataset/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def initiate_training_parameters():
    train_preprocessor = Preprocess(DATASET_PATH + "train (Copy)", scale_size=256, purpose='single_scale_training')
    val_preprocessor = Preprocess(DATASET_PATH + 'val')

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
    


