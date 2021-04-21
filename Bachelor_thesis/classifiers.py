from torch import nn
import torch
import numpy as np
from collections import defaultdict

class Sequential(nn.Sequential):

    def save(self, filename):
        torch.save(self, filename)
