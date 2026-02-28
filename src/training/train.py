import torch
from preprocessing.preprocessing import Preprocess
import torch.optim as optim
import torch.nn as nn
from training.model import VGG11
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Training(nn.Module):
    def __init__(self, train_dataloader, val_dataloader, batch_size, model, epochs, criterion, optimizer, lr_scheduler):
        super(Training, self).__init__()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.model_name = model.__class__.__name__
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler


        self.train_loss = []
        self.val_loss = []

        self.train_accuracy = []
        self.val_accuracy = []

        self.best_loss = float("inf")
        self.best_accuracy = 0.0


    def train_model(self):
        for epoch in tqdm(self.epochs, 'Model Training'):
            self.model.train(True)
            avg_loss = self.train_one_epoch()
            
            running_v_loss = 0.0
            
            self.model.eval()
            
            with torch.no_grad():
                for i, vdata in enumerate(self.val_dataloader):
                    vinputs, vlabels = vdata
                    voutputs = self.model(vinputs)
                    vloss = self.criterion(voutputs, vlabels)
                    running_v_loss += vloss
                    
            avg_vloss = running_v_loss / (i + 1) 
            print('Loss train {} validation {}'.format(avg_loss, avg_vloss))
                       


    def validate_model(self):
        pass

    
    def train_one_epoch(self):
        running_loss = 0. 
        last_loss = 0.
        
        for i, data in enumerate(self.train_dataloader):
            inputs, labels = data
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            self.optimizer.step()
            running_loss += loss.item()
            
            if i % 1000 == 999:
                last_loss = running_loss / 1000
                print(' batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0
        
        return last_loss
            

