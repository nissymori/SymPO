import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(
        self,
        clip_min,
        clip_max,
        dropout_ratio=0.3,
        use_dropout=False,
        use_batchnorm=False,
    ):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.flatten = nn.Flatten()
        self.dropout_ratio = dropout_ratio
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.clip_min = clip_min
        self.clip_max = clip_max

    def forward(self, x, visualize=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        if self.use_dropout:
            x = nn.Dropout(self.dropout_ratio)(x)
        if self.use_batchnorm:
            x = F.relu(self.bn1(self.fc1(x)))
        else:
            x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = nn.Dropout(self.dropout_ratio)(x)
        if self.use_batchnorm:
            x = F.relu(self.bn2(self.fc2(x)))
        else:
            x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = nn.Dropout(self.dropout_ratio)(x)
        output = self.fc3(x)
        output = torch.clamp(output, min=self.clip_min, max=self.clip_max)
        if visualize:
            return output, x.detach().cpu()
        else:
            return output
