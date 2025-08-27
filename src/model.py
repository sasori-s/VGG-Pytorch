from local_response_normalization import LocalResponseNormalization
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
from torch.nn import LocalResponseNorm
from colorama import Fore, Style, init
import time

init(autoreset=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class VGG11(nn.Module):
    def __init__(self, input, num_classes=100):
        super(VGG11, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.reLU = nn.ReLU()

        # No grad else it will be added to backward graph.
        with torch.no_grad():
            flattened_tensor = self._calculating_in_dim_for_linear_layer(input)

        self.fc1 = nn.Linear(flattened_tensor, 4096) #nn.Linear(512 * 7 * 7) which is nn.Linear(last_cnn's channel size, kernel[0], kernel[1], fc1's output size).
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.soft_max = nn.Softmax()


    def forward(self, input):
        x = self.conv1(input)
        x = self.reLU(x)
        x = self.max_pooling(x)

        x = self.conv2(x)
        x = self.reLU(x)
        x = self.max_pooling(x)

        x = self.conv3(x)
        x = self.reLU(x)
        x = self.max_pooling(x)

        x = self.conv4(x)
        x = self.reLU(x)
        x = self.max_pooling(x)

        x = self.conv5(x)
        x = self.reLU(x)
        x = self.max_pooling(x)

        x = self.fc1(x.view(x.size(0), -1))
        x = self.reLU(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.reLU(x)
        x = self.dropout(x)

        x = self.fc3(x)
        probabilities = self.soft_max(x)
        return x
    

    #This funciton is for dynamic way for transitioning CNN to FC
    def _calculating_in_dim_for_linear_layer(self, input):
        conv_layers = nn.Sequential(
            self.conv1, self.reLU, self.max_pooling,
            self.conv2, self.reLU, self.max_pooling,
            self.conv3, self.reLU, self.max_pooling,
            self.conv4, self.reLU, self.max_pooling,
            self.conv5, self.reLU, self.max_pooling
        ).to(DEVICE)

        x = conv_layers(input)

        output = x.numel() // x.shape[0]
        return output


    def initialize_weight_and_bias(self, m):
        if isinstance(m, nn.Conv2d):
            initial_weight = m.weight.mean()
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
            print(f"{Fore.LIGHTCYAN_EX} THe initial weight is {initial_weight} and the new weight is {m.weight.data.mean()}")

        elif isinstance(m, nn.Linear):
            initial_weight = m.weight.mean()
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
            print(f"{Fore.LIGHTCYAN_EX} THe initial weight is {initial_weight} and the new weight is {m.weight.data.mean()}")

    
    def check_weights(self):
        name = self.__class__.__name__
        for name, param in self.named_parameters():
            print(f"{Fore.LIGHTYELLOW_EX} The name of the layer is {name} and the weight is {param.data.mean()}")



if __name__ == '__main__':
    input = torch.randn(32, 3, 224, 224).to(DEVICE)
    model = VGG11(input)
    print(next(model.parameters()).device)
    model.to(DEVICE)
    ouptut = model(input)
    print(summary(model, (3, 224, 224)))
    
    