# student_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class StudentPolicy(nn.Module):
    """
    A smaller architecture for demonstration. Outputs raw logits over actions.
    """
    def __init__(self, in_channels=4, num_actions=4):
        super(StudentPolicy, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(7*7*32, 256)
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits
