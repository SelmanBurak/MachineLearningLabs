import torch
import torch.nn as nn

class SelmanViT(nn.Module):
    """
    8-Head Vision Transformer Architecture
    Input: (B, 1, 28, 28)
    Tokenizer: Conv2d(1, 64, kernel_size=7, stride=7) -> 16 Tokens of dimension 64
    Transformer: 2 TransformerEncoderLayers with 8 heads (dim=64, nhead=8, d_ff=128)
    Classification: LayerNorm -> Mean Token Pooling -> Linear(64, 10)
    """
    def __init__(self, in_channels=1, embed_dim=64, num_heads=8, num_classes=10):
        super().__init__()
        self.tokenizer = nn.Conv2d(in_channels, embed_dim, kernel_size=7, stride=7)
        self.pos_encoding = nn.Parameter(torch.randn(1, 16, embed_dim) * 0.02)
        
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=128,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (B, 1, 28, 28) -> (B, 64, 4, 4)
        tokens = self.tokenizer(x).flatten(2).transpose(1, 2)  # (B, 16, 64)
        tokens = tokens + self.pos_encoding
        out = self.encoder(tokens)
        out = self.ln(out)
        pooled = out.mean(dim=1)
        return self.head(pooled)
