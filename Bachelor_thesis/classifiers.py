from torch import nn
import torch


class Sequential(nn.Sequential):

    def save(self, filename):
        torch.save(self, filename)


class ResidualBlock(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        residual = x

        for layer in self.layers:
            x = layer(x)

        x += residual

        x = nn.ReLU()(x)
        return x


class BasicResidual(ResidualBlock):
    def __init__(self, input_size):
        layers = nn.ModuleList()

        in_s = input_size
        out_s = input_size
        layers.append(nn.Linear(in_s, out_s))
        layers.append(nn.BatchNorm1d(out_s))
        layers.append(nn.ReLU())

        in_s = input_size
        out_s = input_size
        layers.append(nn.Linear(in_s, out_s))
        layers.append(nn.BatchNorm1d(out_s))
        layers.append(nn.ReLU())

        super().__init__(layers)


class FeedForwardNet(nn.Module):
    def __init__(self, input_size, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.layers = nn.ModuleList()

        hidden_layer_size = 128

        in_s = input_size
        out_s = hidden_layer_size
        self.layers.append(nn.Linear(in_s, out_s))
        self.layers.append(nn.ReLU())

        # BEGIN - HIDDEN
        in_s = out_s
        out_s = hidden_layer_size
        self.layers.append(nn.Linear(in_s, out_s))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(out_s))
        self.layers.append(nn.Dropout(p=0.25))

        in_s = out_s
        out_s = hidden_layer_size
        self.layers.append(nn.Linear(in_s, out_s))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(out_s))
        self.layers.append(nn.Dropout(p=0.25))

        in_s = out_s
        out_s = hidden_layer_size
        self.layers.append(nn.Linear(in_s, out_s))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(out_s))
        self.layers.append(nn.Dropout(p=0.25))

        in_s = out_s
        out_s = hidden_layer_size
        self.layers.append(nn.Linear(in_s, out_s))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(out_s))
        self.layers.append(nn.Dropout(p=0.25))

        # END - HIDDEN

        in_s = out_s
        out_s = n_classes
        self.layers.append(nn.Linear(in_s, out_s))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def save(self, filename):
        torch.save(self, filename)

