import torch
import torch.nn as nn
import torch.nn.functional as F

class StudentNet1(nn.Module):
    """
    Student network Dist-KL-net1:
    Input shape: [batch_size, in_channels, 84, 84]
    Output shape: Q-values for each action.
    """
    def __init__(self, in_channels=4, num_actions=4):
        super(StudentNet1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7*7*32, 256)
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class StudentNet2(nn.Module):
    """
    Student network Dist-KL-net2:
    Input shape: [batch_size, in_channels, 84, 84]
    Output shape: Q-values for each action.
    """
    def __init__(self, in_channels=4, num_actions=4):
        super(StudentNet2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7*7*16, 128)
        self.fc2 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class StudentNet3(nn.Module):
    """
    Student network Dist-KL-net3:
    Input shape: [batch_size, in_channels, 84, 84]
    Output shape: Q-values for each action.
    """
    def __init__(self, in_channels=4, num_actions=4):
        super(StudentNet3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7*7*16, 64)
        self.fc2 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
