# teacher_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Classic DQN conv architecture for Atari:
    Input shape: [batch_size, in_channels, 84, 84]
    Output shape: Q-values for each action.
    """
    def __init__(self, in_channels=4, num_actions=4):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
