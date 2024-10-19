import torch
import torch.nn.functional as F

class CNN(torch.nn.Module):
    def __init__(self, 
                 input_size,  # (channels, height, width)
                 base_channels=8, 
                 hidden_units=64, 
                 dropout1_rate=0.25,
                 dropout2_rate=0.5,
                 base_activation=torch.nn.SiLU):
        super(CNN, self).__init__()
        input_channels, height, width = input_size
        self.base_activation = base_activation()
        self.conv1 = torch.nn.Conv2d(input_channels, base_channels, 3, 1)
        self.conv2 = torch.nn.Conv2d(base_channels, base_channels * 2, 3, 1)
        self.conv3 = torch.nn.Conv2d(base_channels * 2, base_channels, 1)
        conv_output_size = self._get_conv_output_size(height, width, base_channels)
        self.fc1 = torch.nn.Linear(conv_output_size, hidden_units)
        self.fc2 = torch.nn.Linear(hidden_units, hidden_units // 4)
        self.fc3 = torch.nn.Linear(hidden_units // 4, 10)
        self.dropout1 = torch.nn.Dropout2d(dropout1_rate)
        self.dropout2 = torch.nn.Dropout(dropout2_rate)

    def _get_conv_output_size(self, height, width, base_channels):
        size_h = (height - 2) // 2
        size_h = (size_h - 2) // 2
        size_w = (width - 2) // 2
        size_w = (size_w - 2) // 2
        return size_h * size_w * base_channels

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
        x = self.base_activation(self.fc2(x))
        x = self.fc3(x)
        return x
