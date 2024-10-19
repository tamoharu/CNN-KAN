import torch
import torch.nn.functional as F
from .kan import KAN
from .cnn import CNN

class CNNKAN(CNN):
    def __init__(self, 
                 input_size,  # (channels, height, width)
                 base_channels=8, 
                 kan_hidden=64,
                 dropout1_rate=0.25,
                 dropout2_rate=0.5,
                 base_activation=torch.nn.SiLU):
        super(CNNKAN, self).__init__(input_size, base_channels, kan_hidden, dropout1_rate, dropout2_rate, base_activation)
        _, height, width = input_size
        conv_output_size = self._get_conv_output_size(height, width, base_channels)
        del self.fc1
        del self.fc2
        del self.fc3

        self.fc1 = torch.nn.Linear(conv_output_size, kan_hidden)
        self.kan = KAN([kan_hidden, kan_hidden // 2, 10])

    def forward(self, x):
        x = self.base_activation(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = self.base_activation(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.base_activation(self.conv3(x))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.base_activation(self.fc1(x))
        x = self.dropout2(x)
        x = self.kan(x)
        return x