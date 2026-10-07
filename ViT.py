"""Minimal Vision Transformer (ViT) from scratch in PyTorch.

Pipeline: image (B,C,H,W)
  -> patchify + linear projection        (B, N, D)      N = (H/p)*(W/p)
  -> prepend CLS token, add pos. embed.  (B, N+1, D)
  -> L x pre-LN Transformer blocks       (B, N+1, D)
  -> LayerNorm, take CLS token -> head   (B, num_classes)

Run `python vit.py` to execute the self-tests.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Split the image into p x p patches and linearly project each to D dims.

    A Conv2d with kernel = stride = p is exactly 'flatten patch + shared Linear'.
    The explicit reshape version is kept in `patchify` to show the equivalence.
    """

    def __init__(self, img_size=32, patch=4, in_ch=3, dim=64):
        super().__init__()
        assert img_size % patch == 0
        self.patch = patch
        self.num_patches = (img_size // patch) ** 2
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x):                          # (B,C,H,W)
        x = self.proj(x)                           # (B,D,H/p,W/p)
        return x.flatten(2).transpose(1, 2)        # (B,N,D): permute data, THEN reshape


def patchify(imgs, p):
    """(B,C,H,W) -> (B, N, C*p*p), patches row-major, each flattened in (C,p,p) order."""
    B, C, H, W = imgs.shape
    x = imgs.reshape(B, C, H // p, p, W // p, p)   # (B,C,gh,p,gw,p)
    x = x.permute(0, 2, 4, 1, 3, 5)                # (B,gh,gw,C,p,p)
    return x.reshape(B, (H // p) * (W // p), C * p * p)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads, attn_drop=0.0):
        super().__init__()
        assert dim % heads == 0
        self.h, self.dk = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)         # one fused projection for Q, K, V
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(attn_drop)

    def forward(self, x):                          # (B,N,D)
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]           # each (B,h,N,dk)
        scores = q @ k.transpose(-2, -1) / self.dk ** 0.5   # (B,h,N,N)
        attn = self.drop(scores.softmax(dim=-1))   # softmax is max-subtracted internally
        out = attn @ v                             # (B,h,N,dk)
        out = out.transpose(1, 2).reshape(B, N, D) # NOT .view: non-contiguous after transpose
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class Block(nn.Module):
    """Pre-LN transformer block: x + MHSA(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, dim, heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, drop)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    def __init__(self, img_size=32, patch=4, in_ch=3, num_classes=10,
                 dim=64, depth=4, heads=4, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch, in_ch, dim)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, dim))   # learned, as in ViT
        self.pos_drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):                                  # (B,C,H,W)
        x = self.patch_embed(x)                            # (B,N,D)
        cls = self.cls_token.expand(x.shape[0], -1, -1)    # (B,1,D), expand = no copy
        x = torch.cat([cls, x], dim=1) + self.pos_embed    # (B,N+1,D)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])                          # classify from CLS token


# ----------------------------------------------------------------------------- tests
if __name__ == "__main__":
    torch.manual_seed(0)

    # 1. Conv patch embedding == patchify + Linear with the same weights
    pe = PatchEmbed(32, 4, 3, 64)
    x = torch.randn(2, 3, 32, 32)
    lin = patchify(x, 4) @ pe.proj.weight.reshape(64, -1).T + pe.proj.bias
    assert torch.allclose(pe(x), lin, atol=1e-5)

    # 2. our MHSA matches torch.nn.MultiheadAttention with copied weights
    D, H = 64, 4
    mine = MultiHeadSelfAttention(D, H)
    ref = nn.MultiheadAttention(D, H, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(mine.qkv.weight)
        ref.in_proj_bias.copy_(mine.qkv.bias)
        ref.out_proj.weight.copy_(mine.proj.weight)
        ref.out_proj.bias.copy_(mine.proj.bias)
    t = torch.randn(2, 9, D)
    assert torch.allclose(mine(t), ref(t, t, t, need_weights=False)[0], atol=1e-5)

    # 3. end-to-end shape + it can overfit a tiny batch (sanity check that gradients flow)
    model = ViT(img_size=32, patch=4, num_classes=10, dim=64, depth=2, heads=4)
    imgs, labels = torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))
    assert model(imgs).shape == (8, 10)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    first = None
    for step in range(100):
        loss = F.cross_entropy(model(imgs), labels)
        opt.zero_grad(); loss.backward(); opt.step()
        first = first if first is not None else loss.item()
    print(f"overfit check: loss {first:.3f} -> {loss.item():.4f}")
    assert loss.item() < 0.05 * first
    print("params:", sum(p.numel() for p in model.parameters()))
    print("all ViT tests passed")
