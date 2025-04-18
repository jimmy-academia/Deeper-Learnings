import torch

class GoodModel(object):
    """docstring for GoodModel"""
    def __init__(self, arg):
        super(GoodModel, self).__init__()
        self.arg = arg
        self.linear = torch.nn.Sequential(
                torch.nn.Conv2d(),
                torch.nn.ReLU()
            )

    def forward(self, x):
        return self.linear(x)