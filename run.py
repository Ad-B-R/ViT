from data import get_cifar10
import torch
import torch.nn as nn
import model
import json

class ViT(nn.Module):
    def __init__(self, f, *args, **kwargs):
        super().__init__(*args, **kwargs)
        