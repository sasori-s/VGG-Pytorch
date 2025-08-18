import torch
from preprocessing import Preprocess
import torch.optim as optim
import torch.nn as nn
from model import VGG11
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LRScheduler


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Training(nn.Module):
    def __init__(self, dataloader, batch_size, model, epochs, criterion, optimizer, scheduler, data_path):
        self.dataloader = dataloader
        self.model = model
        self.model_name = model.__class__.__name__
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler(self.optimizer)


        self.train_loss = []
        self.val_loss = []

        self.train_accuracy = []
        self.val_accuracy = []

        self.best_loss = float("inf")
        self.best_accuracy = 0.0


    def train_model(self):
        pass


    def validate_model(self):
        pass

    
    def training_process(self):
        self.training_data = self.dataloader()

