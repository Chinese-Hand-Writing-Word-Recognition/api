import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.AvgPool2d(kernel_size, stride, padding)
        # torch.nn.AdaptiveAvgPool2d = keras' GlobalAvgPooling

        # input image size: [3, 96, 96]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(3, 2, 0),

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
            nn.Dropout(0.8)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(448, 801)
        )
    def forward(self, x):
        # input (x): [batch_size, 3, 96, 96]
        # output: [batch_size, 801]

        # Extract features by convolutional layers.
        x = self.cnn_layers(x)
        # The extracted feature map must be flatten before going to fully-connected layers.
        x = x.flatten(1)
        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x