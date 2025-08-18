from model import VGG11
from train import Training
from preprocessing import Preprocess
from precompute_mean_std import PreCompute
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import LRScheduler

DATASET_PATH = "/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/CIFAR-100-dataset/"

def initiate_training_parameters():
    train_preprocessor = Preprocess(DATASET_PATH + "train", scale_size=256, purpose='training')
    val_preprocessor = Preprocess(DATASET_PATH + 'test')

    train_dataloader = train_preprocessor()

    criterion = nn.CrossEntropyLoss()

    model = VGG11(224, num_classes=100)
    optimizer_sgd = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=5 * 10e-4, momentum=0.9)
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-2)
    lr_scheduler = LRScheduler(optimizer_sgd)
    epochs = 50
    batch_size = 64

    model_kwgs = {
        "model" : model,
        "dataloader": train_dataloader,
        "epochs" : epochs,
        "criterion" : criterion,
        "optimizer" : optimizer_sgd,
        "lr_scheduler" : lr_scheduler,
        "batch_size" : batch_size,

    } 

    return model_kwgs
    

def main():
    kwgs = initiate_training_parameters()
    trainer = Training(**kwgs)
    

if __name__ == '__main__':
    main()
    


