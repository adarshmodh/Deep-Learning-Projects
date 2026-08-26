import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

class TransformerBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        # Instantiate in this order:
        # 1. self.MultiHeadedSelfAttention(model_dim, num_heads)
        # 2. self.VanillaNeuralNetwork(model_dim)
        # 3. Two nn.LayerNorm(model_dim) instances
        self.attention = MultiHeadedSelfAttention(model_dim, num_heads)
        self.linear_network = VanillaNeuralNetwork(model_dim)
        self.first_norm = nn.LayerNorm(model_dim)
        self.second_norm = nn.LayerNorm(model_dim)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # Two residual connections with Pre-LN:
        #   x = x + attention(layer_norm_1(x))
        #   x = x + feed_forward(layer_norm_2(x))
        # Return result rounded to 4 decimal places
        embedded = embedded + self.attention(self.first_norm(embedded)) # skip connection
        embedded = embedded + self.linear_network(self.second_norm(embedded)) # another skip connection
        return torch.round(embedded, decimals=4)

class MultiHeadedSelfAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        # Create num_heads SingleHeadAttention instances using nn.ModuleList
        # Each head size = attention_dim // num_heads
        # Use: self.SingleHeadAttention(embedding_dim, head_size)
        # After the heads, add an output projection: nn.Linear(attention_dim, attention_dim, bias=False)
        head_size = attention_dim // num_heads
        self.heads = nn.ModuleList([SingleHeadAttention(embedding_dim, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(attention_dim, attention_dim, bias=False)
        
    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # Run each head on the input, concatenate outputs along dim=2
        # Pass concatenated result through the output projection (W_O)
        # Return result rounded to 4 decimal places
        out = torch.cat([h(embedded) for h in self.heads], dim=-1)
        out = self.proj(out)
        return torch.round(out, decimals=4)

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

        return attn

class VanillaNeuralNetwork(nn.Module):
        def __init__(self, model_dim: int):
            super().__init__()
            torch.manual_seed(0)
            self.up_projection = nn.Linear(model_dim, model_dim * 4)
            self.relu = nn.ReLU()
            self.down_projection = nn.Linear(model_dim * 4, model_dim)
            self.dropout = nn.Dropout(0.2) # using p = 0.2

        def forward(self, x: TensorType[float]) -> TensorType[float]:
            torch.manual_seed(0)
            return self.dropout(self.down_projection(self.relu(self.up_projection(x))))

