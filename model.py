import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PatchInputEmbedding(nn.Module):
    def __init__(self, height, width, channel, patch, **kwargs):
        super().__init__(**kwargs)
        self.height = height
        self.width = width
        self.channel = channel
        self.patch = patch

class PositionalEncoding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, seq_len, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def attention(query, key, value, dropout):
        d_k = query.shape[-1]
        attention_score = (query@key.T)/(math.sqrt(d_k))

    def forward():
        pass

class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.beta = nn.Parameter(torch.zeros(features))
        self.gamma = nn.Parameter(torch.ones(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.gamma*(x-mean)/std + self.gamma
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ResidualConnection(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ViT(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)