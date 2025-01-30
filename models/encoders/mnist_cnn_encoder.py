import torch.nn as nn

class MnistCnnEncoder(nn.Module):
    def __init__(self):
        super(MnistCnnEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(32*7*7,256),
            nn.Tanh()
        )
        
    def forward(self, x):
         B,C,W,H = x.shape
         assert C == 1 and W == 28 and H == 28, 'input tensor should be of dimension B*1*28*28'
         x = self.conv1(x)
         x = self.conv2(x)
         x = x.view(B,-1)
         x = self.fc1(x)
         return x