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
        x /= 255.0
        x = self.cnn(x)  # Apply CNN layers
        x = self.linear(x)  # Apply Linear layers
        return x

class QNetwork(nn.Module):
    def __init__(self, conv1_out_channels=32, conv2_out_channels=64, conv3_out_channels=64, linear_out_features=512):
        super(QNetwork, self).__init__()
        self.features_extractor = NatureCNN(conv1_out_channels, conv2_out_channels, conv3_out_channels, linear_out_features)
        
        # Q-value estimation network
        self.q_net = nn.Sequential(
            nn.Linear(linear_out_features, 4)  # Output 4 Q-values (for 4 possible actions)
        )
    
    def forward(self, x):
        x = self.features_extractor(x)  # Extract features using CNN and linear layers
        q_values = self.q_net(x)  # Get Q-values for each action
        return q_values
    
# Define QNetworkSmall that extends QNetwork with smaller parameters
class QNetworkSmall(QNetwork):
    def __init__(self):
        # Initialize with smaller parameters
        super(QNetworkSmall, self).__init__(conv1_out_channels=16, 
                                            conv2_out_channels=32, 
                                            conv3_out_channels=32, 
                                            linear_out_features=256)

if __name__ == "__main__":
    # Create the original model
    original_model = QNetwork()
    print("Original Model:")
    print(original_model)

    # Create a smaller version with roughly half the parameters
    small_model = QNetwork(conv1_out_channels=16, conv2_out_channels=32, conv3_out_channels=32, linear_out_features=256)
    print("\nSmall Model:")
    print(small_model)
