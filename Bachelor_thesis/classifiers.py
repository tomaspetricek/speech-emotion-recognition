from torch import nn
import torch


class Sequential(nn.Sequential):

    def save(self, filename):
        torch.save(self, filename)


class FeedForwardNet(nn.Module):
    def __init__(self, input_size, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.layers = nn.ModuleList()

        in_s = input_size
        out_s = 128
        self.layers.append(nn.Linear(in_s, out_s))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(out_s))

        in_s = out_s
        out_s = 128
        self.layers.append(nn.Linear(in_s, out_s))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(out_s))

        in_s = out_s
        out_s = 128
        self.layers.append(nn.Linear(in_s, out_s))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(out_s))

        in_s = out_s
        out_s = 64
        self.layers.append(nn.Linear(in_s, out_s))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(out_s))

        in_s = out_s
        out_s = n_classes
        self.layers.append(nn.Linear(in_s, out_s))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def save(self, filename):
        torch.save(self, filename)

