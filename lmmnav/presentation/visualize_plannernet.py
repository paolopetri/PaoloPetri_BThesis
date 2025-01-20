import torch
import visualkeras
from PIL import ImageFont
import torch.nn as nn

# Define a simplified block to stand in for PerceptNet encoders
def simplified_encoder(in_channels, out_channels, kernel_size=7, stride=2, padding=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # You can add more layers to mimic deeper features if desired
    )

# Define a simplified decoder block
def simplified_decoder(in_channels, goal_channels, k=5):
    return nn.Sequential(
        nn.Conv2d(in_channels + goal_channels, 512, kernel_size=5, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(256 * 128, 1024),  # Adjust input features if needed
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, k * 3)  # Final output layer for trajectory points
        # Note: This simple decoder doesn't replicate the exact behavior of your original decoder.
    )

# Create the simplified PlannerNet-like structure using Sequential
class SimplePlannerNet(nn.Module):
    def __init__(self, encoder_channels=64, goal_channels=64, k=5):
        super(SimplePlannerNet, self).__init__()
        # Simplified encoders for depth and risk paths
        self.encoder1 = simplified_encoder(3, encoder_channels)
        self.encoder2 = simplified_encoder(3, encoder_channels)
        
        # For concatenation simulation, assume outputs are same shape
        # Simplify channel dimensions after concatenation: 2 * encoder_channels
        concatenated_channels = 2 * encoder_channels
        
        # Simplified decoder that uses concatenated features plus goal info
        self.decoder = simplified_decoder(concatenated_channels, goal_channels, k)
        
    def forward(self, depth, risk, goal):
        # Pass through simplified encoders
        encoded_risk = self.encoder1(risk)
        encoded_depth = self.encoder2(depth)
        
        # Concatenate along the channel dimension
        x = torch.cat((encoded_depth, encoded_risk), dim=1)
        
        # For simplicity, ignore goal integration in this dummy forward
        # In a real model, you would integrate 'goal' into the decoder as needed
        output = self.decoder(x)
        
        return output

# Instantiate the simple model
simple_model = SimplePlannerNet()

# Print the simple model structure
print(simple_model)


font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
font = ImageFont.truetype(font_path, 12)

visualkeras.layered_view(simple_model, to_file='plannernet.png', font=font, legend=True)