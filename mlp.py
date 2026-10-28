"""MNIST için sıfırdan, yalnız tam bağlı MLP."""
from torch import nn


class MLP(nn.Module):
    def __init__(self, hidden_sizes=(128, 64)):
        super().__init__()
        if len(hidden_sizes) < 2 or any(type(n) is not int or n <= 0 for n in hidden_sizes):
            raise ValueError('En az iki pozitif tamsayı hidden size gerekir.')
        layers = [nn.Flatten()]
        previous = 784
        for width in hidden_sizes:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        layers.append(nn.Linear(previous, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
