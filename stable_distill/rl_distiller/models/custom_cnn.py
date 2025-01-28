"""
Example custom CNN model for stable-baselines3. 
We define a custom feature extractor for the policy.
"""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # Just a minimal example:
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        # calculate shape
        with torch.no_grad():
            sample_input = torch.zeros((1, *observation_space.shape))
            n_flatten = self.cnn(sample_input).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
