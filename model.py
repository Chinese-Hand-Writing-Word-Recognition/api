import torch.nn as nn
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.AvgPool2d(kernel_size, stride, padding)
        # torch.nn.AdaptiveAvgPool2d = keras' GlobalAvgPooling

        # input image size: [3, 128, 128]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(3, 2, 0),
            nn.Dropout(0.5),

            nn.Conv2d(64, 96, 3, 1, 1, bias = False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 64, 3, 1, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 96, 3, 1, 1, bias = False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.AvgPool2d(3, 2, 0),
            nn.Dropout(0.5),

            nn.Conv2d(96, 128, 3, 1, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 96, 3, 1, 1, bias = False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 128, 3, 1, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(3, 2, 0),
            nn.Dropout(0.5),

            nn.Conv2d(128, 256, 3, 1, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 192, 3, 1, 1, bias = False),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 256, 3, 1, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(3, 2, 0),
            nn.Dropout(0.5),

            nn.Conv2d(256, 448, 3, 1, 1, bias = False),
            nn.BatchNorm2d(448),
            nn.ReLU(),
            nn.Conv2d(448, 256, 3, 1, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 448, 3, 1, 1, bias = False),
            nn.BatchNorm2d(448),
            nn.ReLU(),
            nn.AvgPool2d(3, 2, 0),

            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.85)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(448, 801)
        )
    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 801]

        # Extract features by convolutional layers.
        x = self.cnn_layers(x)
        # The extracted feature map must be flatten before going to fully-connected layers.
        x = x.flatten(1)
        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x


import torch
from torch import nn
from torchsummary import summary
from efficientnet_pytorch import EfficientNet

import config

class EfNetModel(nn.Module):
    def __init__(self, num_classes=801, dropout=0.2, pretrained_path=""):
        super().__init__()
        
        self.model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)
        if not config.multi_channel:
            self.model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)
        else:
            self.model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes, in_channels=6)

        if pretrained_path:
            self.load_state_dict(torch.load(pretrained_path))
        
        # set dropout
        self.model._dropout = nn.Dropout(dropout)
        
        for param in self.model.parameters():
            param.requires_grad = True

        # for param in self.model._fc.parameters():
        #     param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x


