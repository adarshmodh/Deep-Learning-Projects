import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Q: (batch_size, Q_seq_len, d_out)
    K: (batch_size, K_seq_len, d_out)
    V: (batch_size, V_seq_len, d_out)
    mask: (batch_size, K_seq_len)
    """
    B, T, C = Q.shape
    attn = Q @ K.transpose(-2,-1) * C**-0.5
    if mask != None:
        attn_mask = mask.bool().unsqueeze(1)
        attn = attn.masked_fill(attn_mask, float('-inf'))
    attn = F.softmax(attn, dim=-1)
    out = attn @ V
    return out

class SelfAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super(SelfAttention, self).__init__()
        self.key = nn.Linear(d_in, d_out, bias = False)
        self.query = nn.Linear(d_in, d_out, bias = False)
        self.value = nn.Linear(d_in, d_out, bias = False)
        

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_in)
        mask: (batch_size, seq_len)

        Output: (batch_size, seq_len, d_out)
        """
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_mat = scaled_dot_product_attention(Q, K, V, mask)
        return attn_mat




############TESTING CODE 

import torch
from attention import SelfAttention, scaled_dot_product_attention


def test_scaled_dot_product_attention_shape():
    """Test that attention output has correct shape."""
    batch_size, q_seq_len, k_seq_len, d_model = 2, 10, 15, 64
    Q = torch.randn(batch_size, q_seq_len, d_model)
    K = torch.randn(batch_size, k_seq_len, d_model)
    V = torch.randn(batch_size, k_seq_len, d_model)
    
    out = scaled_dot_product_attention(Q, K, V)
    
    assert out.shape == (batch_size, q_seq_len, d_model), f"Expected {(batch_size, q_seq_len, d_model)}, got {out.shape}"
    print("✓ scaled_dot_product_attention: output shape correct")


def test_scaled_dot_product_attention_masking():
    """Test that masked positions get zero attention weight  for a single sequence (batch_size=1)."""
    batch_size, seq_len, d_model = 1, 4, 8
    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.zeros(batch_size, seq_len, d_model)
    # Set V[0, 0] to ones - if masking works, this should not appear in output
    V[0, 0, :] = 1.0
    
    # Mask out the first key position
    mask = torch.tensor([[0, 1, 1, 1]])  # First position masked
    
    out = scaled_dot_product_attention(Q, K, V, mask)
    
    # If masking works, position 0 of V (all ones) should contribute nothing
    # So output should be all zeros (since other V positions are zeros)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6), "Masked position should not contribute to output"
    print("✓ scaled_dot_product_attention: single-batch masking works correctly")

def test_scaled_dot_product_attention_masking_batched():
    """Test that masked positions get zero attention weight when batch_size > 1."""
    batch_size, seq_len, d_model = 2, 4, 8
    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.zeros(batch_size, seq_len, d_model)
    
    # Setup V so that the ONLY non-zero values are at specific index targets
    V[0, 0, :] = 1.0   # Batch 0: first position has data
    V[1, -1, :] = 1.0  # Batch 1: last position has data
    
    # Mask out EXACTLY those target positions
    # 0 means masked out (ignored), 1 means keep
    mask = torch.tensor([
        [0, 1, 1, 1],  # Batch 0: Mask out the first position
        [1, 1, 1, 0]   # Batch 1: Mask out the last position
    ])
    
    # Run attention
    out = scaled_dot_product_attention(Q, K, V, mask)
    
    # If the batched masking aligns correctly, the attention mechanism 
    # will assign 0 weight to the only places in V that have data.
    # Therefore, the entire output tensor should be zeros.
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6), "Masked positions leaked into output for batched inputs!"
    print("✓ scaled_dot_product_attention: batched masking works correctly")


def test_self_attention_shape():
    """Test SelfAttention output shape."""
    batch_size, seq_len, d_in, d_out = 2, 10, 64, 32
    
    model = SelfAttention(d_in, d_out)
    x = torch.randn(batch_size, seq_len, d_in)
    
    out = model(x)
    
    assert out.shape == (batch_size, seq_len, d_out), f"Expected {(batch_size, seq_len, d_out)}, got {out.shape}"
    print("✓ SelfAttention: output shape correct")


def test_self_attention_with_mask():
    """Test SelfAttention with masking."""
    batch_size, seq_len, d_in, d_out = 2, 5, 16, 8
    
    model = SelfAttention(d_in, d_out)
    x = torch.randn(batch_size, seq_len, d_in)
    mask = torch.ones(batch_size, seq_len)
    mask[:, -2:] = 0  # Mask last 2 positions
    
    out = model(x, mask)
    
    assert out.shape == (batch_size, seq_len, d_out)
    assert not torch.isnan(out).any(), "Output contains NaN values"
    print("✓ SelfAttention: masking runs without errors")


if __name__ == "__main__":
    print("Running attention module tests...\n")
    
    test_scaled_dot_product_attention_shape()
    test_scaled_dot_product_attention_masking()
    test_scaled_dot_product_attention_masking_batched()
    test_self_attention_shape()
    test_self_attention_with_mask()
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
