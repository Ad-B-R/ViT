import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PatchInputEmbedding(nn.Module):
    def __init__(self, height, width, d_model, channel, patch, **kwargs):
        super().__init__(**kwargs)
        self.height = height
        self.width = width
        self.channel = channel
        self.patch = patch

        self.num_patches = width*height//(patch**2)
        self.patch_dim = channel*(patch**2)

        self.unfold = nn.Unfold(kernel_size=self.patch, stride=self.patch)
        self.proj_layer = nn.Linear(self.patch_dim, d_model)
        
        self.cls_token = nn.Parameter(torch.empty(1,1,d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        
        self.pos_embedding = nn.Parameter(torch.empty(1, self.num_patches + 1, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)
    
    def forward(self, x):
        x = self.unfold(x)
        x = x.transpose(-1, -2)

        x = self.proj_layer(x)
        broadcasted_cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((broadcasted_cls, x), dim=1)
        return x + self.pos_embedding

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.h = h
        self.d_k = d_model//h

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)

        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, dropout):
        d_k = query.shape[-1]
        attention_score = (query@key.transpose(-2,-1))/(math.sqrt(d_k))

        attention_score = attention_score.softmax(dim=-1)

        return attention_score @ value, attention_score

    def forward(self, x):
        # batch, seq_len, d_model
        query = self.Wq(x)
        key = self.Wk(x)
        value = self.Wv(x)
        
        # RESHAPE       

        query = query.view(query.shape[0], -1, self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], -1, self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], -1, self.h, self.d_k).transpose(1,2)

        x, _ = self.attention(query=query, key=key, value=value, dropout=self.dropout)
        x = x.transpose(1,2)

        x = x.contiguous().view(x.shape[0], -1, self.d_model)
        
        return self.Wo(x)        

class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.beta = nn.Parameter(torch.zeros(features))
        self.gamma = nn.Parameter(torch.ones(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.gamma*(x-mean)/std + self.beta
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.ffn1 = nn.Linear(d_model, d_ff)
        self.ffn2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.ffn2(self.dropout(F.relu(self.ffn1(x))))

class ResidualConnection(nn.Module):
    def __init__(self, features, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = LayerNorm(features)

    def forward(self, x, sublayer:nn.ModuleList):
        return x + (self.dropout(sublayer(self.layernorm(x))))

class Encoder(nn.Module):
    def __init__(self, features, dropout, attn, ffn, **kwargs):
        super().__init__(**kwargs)
        self.attention_block = attn
        self.ffn = ffn
        self.residual_connections = nn.ModuleList(ResidualConnection(features, dropout) for _ in range(2))

    def forward(self, x):
        x = self.residual_connections[0](x, lambda x: self.attention_block(x))
        x = self.residual_connections[1](x, lambda x: self.ffn(x))
        return x

class ViT_Block(nn.Module):
    def __init__(self, embed, layers, encoder, d_model, num_classes, dropout, **kwargs):
        super().__init__(**kwargs)
        self.embed = embed
        self.encoders = encoder
        self.d_model = d_model
        self.layers = layers
        
        self.layernorm = LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.dropout(self.embed(x))
        for encoder in self.encoders:
            x = encoder(x)
        
        x = self.layernorm(x)
        x = x[:,0,:]
        x = self.classifier(x)
        return x

