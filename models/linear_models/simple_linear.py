import torch.nn as nn

class SimpleLinear(nn.Module):
    def __init__(self, input_dim, output_dim, activation=None):
        super(SimpleLinear, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.activation = activation
        
    def forward(self, x):
        x = self.fc1(x)
        if self.activation is not None:
            x = self.activation(x)
        return x