import torch
from preprocessing import Preprocess
import torch.optim as optim
import torch.nn as nn
from model import VGG11
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LRScheduler


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Training(nn.Module):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.model = VGG11(224 ,num_classes=100)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_sgd = torch.optim.SGD(self.model.parameters(), lr=1e-2, weight_decay=5 * 10e-4, momentum=0.9)
        self.optimizer_adam = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        self.scheduler = LRScheduler(self.optimizer)

