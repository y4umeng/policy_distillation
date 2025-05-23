import torch
import torch.nn as nn
import torch.nn.functional as F

class NatureCNN(nn.Module):
    def __init__(self, conv1_out_channels=32, conv2_out_channels=64, conv3_out_channels=64, linear_out_features=512):
        super(NatureCNN, self).__init__()
        
        # Convolutional layers
        self.cnn = nn.Sequential(
            nn.Conv2d(4, conv1_out_channels, kernel_size=(8, 8), stride=(4, 4)),  # Conv1
            nn.ReLU(),
            nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size=(4, 4), stride=(2, 2)),  # Conv2
            nn.ReLU(),
            nn.Conv2d(conv2_out_channels, conv3_out_channels, kernel_size=(3, 3), stride=(1, 1)),  # Conv3
            nn.ReLU(),
            nn.Flatten(start_dim=1)  # Flattening the output
        )
        
        # Linear layer
        self.linear = nn.Sequential(
            nn.Linear(conv3_out_channels * 7 * 7, linear_out_features),  # Assuming the output of Conv3 is (64, 7, 7)
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.cnn(x)  # Apply CNN layers
        x = self.linear(x)  # Apply Linear layers
        return x

class QNetwork(nn.Module):
    def __init__(self, conv1_out_channels=32, conv2_out_channels=64, conv3_out_channels=64, linear_out_features=512, num_actions=3):
        super(QNetwork, self).__init__()
        self.features_extractor = NatureCNN(conv1_out_channels, conv2_out_channels, conv3_out_channels, linear_out_features)
        
        # Q-value estimation network
        self.q_net = nn.Sequential(
            nn.Linear(linear_out_features, num_actions)  # Output num_actions Q-values
        )
    
    def forward(self, x):
        x = torch.div(x, 255.0) # normalize to [0, 1]
        x = self.features_extractor(x)  # Extract features using CNN and linear layers
        q_values = self.q_net(x)  # Get Q-values for each action
        return q_values
    
# 25% size QNetwork, equivalent to Dist-KL-net1 from Policy Distillation paper
class QNetwork1(QNetwork):
    def __init__(self, num_actions):
        # Initialize with smaller parameters
        super(QNetwork1, self).__init__(conv1_out_channels=16, 
                                            conv2_out_channels=32, 
                                            conv3_out_channels=32, 
                                            linear_out_features=256,
                                            num_actions=num_actions)
        
# 7% size QNetwork, equivalent to Dist-KL-net2 from Policy Distillation paper
class QNetwork2(QNetwork):
    def __init__(self, num_actions):
        # Initialize with smaller parameters
        super(QNetwork2, self).__init__(conv1_out_channels=16, 
                                            conv2_out_channels=16, 
                                            conv3_out_channels=16, 
                                            linear_out_features=128,
                                            num_actions=num_actions)
        
# 4% QNetwork, equivalent to Dist-KL-net3 from Policy Distillation paper
class QNetwork3(QNetwork):
    def __init__(self, num_actions):
        # Initialize with smaller parameters
        super(QNetwork3, self).__init__(conv1_out_channels=16, 
                                            conv2_out_channels=16, 
                                            conv3_out_channels=16, 
                                            linear_out_features=64,
                                            num_actions=num_actions)
        
# 1% QNetwork
class QNetwork4(QNetwork):
    def __init__(self, num_actions):
        # Initialize with smaller parameters
        super(QNetwork4, self).__init__(conv1_out_channels=8, 
                                            conv2_out_channels=8, 
                                            conv3_out_channels=8, 
                                            linear_out_features=32,
                                            num_actions=num_actions)
        
# .?% QNetwork
class QNetwork5(QNetwork):
    def __init__(self, num_actions):
        # Initialize with smaller parameters
        super(QNetwork5, self).__init__(conv1_out_channels=4, 
                                            conv2_out_channels=4, 
                                            conv3_out_channels=4, 
                                            linear_out_features=16,
                                            num_actions=num_actions)
# .?% QNetwork
class QNetwork6(QNetwork):
    def __init__(self, num_actions):
        # Initialize with smaller parameters
        super(QNetwork6, self).__init__(conv1_out_channels=2, 
                                            conv2_out_channels=2, 
                                            conv3_out_channels=2, 
                                            linear_out_features=8,
                                            num_actions=num_actions)

def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Create the original model
    print(f"DQN: {count_params(QNetwork(4))} parameters.")
    print(f"DQN1: {count_params(QNetwork1(4))} parameters.")
    print(f"DQN2: {count_params(QNetwork2(4))} parameters.")
    print(f"DQN3: {count_params(QNetwork3(4))} parameters.")
    print(f"DQN4: {count_params(QNetwork4(4))} parameters.")
    print(f"DQN5: {count_params(QNetwork5(4))} parameters.")
    print(f"DQN6: {count_params(QNetwork6(4))} parameters.")