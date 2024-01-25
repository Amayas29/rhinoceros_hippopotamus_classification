import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(ConvNet, self).__init__()

        # We first define the convolution and pooling layers as a features extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channel, 16, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0),
            nn.Dropout(p=0.3),
            nn.Conv2d(16, 32, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0),
            nn.Dropout(p=0.3),
            nn.Conv2d(32, 32, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0),
        )

        # We then define fully connected layers as a classifier
        self.classifier = nn.Sequential(
            nn.Linear(25088, 10000),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(10000, 5000),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(5000, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, n_classes),
        )

    # Method called when we apply the network to an input batch
    def forward(self, X):
        bsize = X.size(0)  # batch size
        output = self.features(X)  # output of the conv layers
        output = output.view(
            bsize, -1
        )  # we flatten the 2D feature maps into one 1D vector for each input
        output = self.classifier(output)  # we compute the output of the fc layers
        return output
