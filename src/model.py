import torch.nn as nn

# Make a neural network
class Recognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(784, 256)
        self.layer2 = nn.Linear(256, 10)

        self.ReLU = nn.ReLU()
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.ReLU(x)
        x = self.layer2(x)

        return(x)