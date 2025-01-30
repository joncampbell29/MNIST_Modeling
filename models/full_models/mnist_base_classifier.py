import torch.nn as nn
from models.encoders.mnist_cnn_encoder import MnistCnnEncoder
from models.linear_models.simple_linear import SimpleLinear


class MnistBaseClassifier(nn.Module):
    def __init__(self):
        super(MnistBaseClassifier, self).__init__()
        self.encoder = MnistCnnEncoder()
        self.model = SimpleLinear(256, 10)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.model(x)
        return x
        