import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False) 
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)
       
    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        
        B,T,C = embedded.shape
        # C is embedding dim 
        # T is context length 

        K = self.key(embedded) # (B, T, attention_dim)
        Q = self.query(embedded) # (B, T, attention_dim)
        V = self.value(embedded) # (B, T, attention_dim)
        
        attn_dim = K.shape[2]
        scores = (Q @ K.transpose(-2,-1)) / attn_dim**0.5
        tril = torch.tril(torch.ones(T, T))
        scores = scores.masked_fill(tril == 0, float('-inf'))   
        scores = F.softmax(scores, dim=-1) #softmax across the context length dimension
        attn = scores @ V

        return torch.round(attn, decimals = 4)
