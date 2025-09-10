import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    """
    Embed raw EEG signals (time‐domain) into token embeddings.
    Input shape: (batch, 1, n_channels, n_times)
    """
    def __init__(self, emb_size: int = 40):
        super().__init__()
        # 1×15 conv over time, then 55×1 conv over channels

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=(1, 15), stride=(1, 1)),   # over time
            nn.Conv2d(40, 40, kernel_size=(55, 1), stride=(1, 1)),  # across all channels
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 20), stride=(1, 5)),       # pool over time
            nn.Dropout(0.5),
        )
        # project 40 → emb_size, then flatten spatial dims into sequence
        # ...existing code...
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, kernel_size=(1, 1), stride=(1, 1)),
            Rearrange('b e h w -> b (h w) e')  # h=1, w=number of time patches
            )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, n_channels, n_times)
        x = self.shallownet(x)      # → (batch, 40, 1, w)
        x = self.projection(x)      # → (batch, w, emb_size)
        return x                    # (batch, num_time_patches, emb_size)


class ChannelFreqEmbedding(nn.Module):
    """
    Project per-channel band-power vectors (freq-domain) into token embeddings.
    Input shape: (batch, n_channels, n_bands)
    Output shape: (batch, n_channels, emb_size)
    """
    def __init__(self, n_bands: int = 5, emb_size: int = 40):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_bands, 32),
            nn.GELU(),
            nn.Linear(32, emb_size)
        )

    def forward(self, x_freq: torch.Tensor) -> torch.Tensor:
        # x_freq shape: (batch, n_channels, n_bands)
        b, ch, nb = x_freq.shape
        out = self.mlp(x_freq.reshape(b * ch, nb))  # → (b*ch, emb_size)
        return out.view(b, ch, -1)                  # → (batch, n_channels, emb_size)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys    = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values  = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # x shape: (batch, seq_len, emb_size)
        queries = rearrange(self.queries(x),  "b n (h d) -> b h n d", h=self.num_heads)
        keys    = rearrange(self.keys(x),     "b n (h d) -> b h n d", h=self.num_heads)
        values  = rearrange(self.values(x),   "b n (h d) -> b h n d", h=self.num_heads)

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** 0.5
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)

        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int, drop_p: float):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int,
                 num_heads: int = 10,
                 drop_p: float = 0.5,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p
                ),
                nn.Dropout(drop_p)
            ))
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int, emb_size: int):
        super().__init__(*[
            TransformerEncoderBlock(emb_size) for _ in range(depth)
        ])


class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int, n_classes: int = 2):
        super().__init__()
        # We'll pool over the sequence dimension with a mean and then do an MLP
        self.norm = nn.LayerNorm(emb_size)
        self.pool = lambda x: reduce(x, 'b n e -> b e', reduction='mean')
        self.fc   = None
        self.n_classes = n_classes

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch, seq_len, emb_size)
        pooled = self.pool(x)            # → (batch, emb_size)
        pooled = self.norm(pooled)
        # Build MLP on first pass:
        if self.fc is None:
            in_fe = pooled.size(1)
            self.fc = nn.Sequential(
                nn.Linear(in_fe, 128),
                nn.ELU(),
                nn.Dropout(0.5),
                nn.Linear(128, 32),
                nn.ELU(),
                nn.Dropout(0.3),
                nn.Linear(32, self.n_classes)
            ).to(pooled.device)
        out = self.fc(pooled)            # → (batch, n_classes)
        return out


class ConformerWithFreq(nn.Module):
    """
    Conformer that consumes both time‐domain patches (via PatchEmbedding)
    and per‐channel frequency‐domain vectors (via ChannelFreqEmbedding).
    """
    def __init__(self,
                 n_channels: int,
                 n_times: int,
                 n_bands: int = 5,
                 emb_size: int = 40,
                 depth: int = 6,
                 n_classes: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        # Embedding for raw time‐series (input shape: (batch, 1, n_ch, n_times))
        self.patch_embed = PatchEmbedding(emb_size=emb_size)

        # Embedding for per‐channel band‐power (input shape: (batch, n_ch, n_bands))
        self.freq_embed  = ChannelFreqEmbedding(n_bands=n_bands, emb_size=emb_size)

        # Transformer stack
        self.transformer = TransformerEncoder(depth=depth, emb_size=emb_size)

        # Classification head
        self.classifier  = ClassificationHead(emb_size, n_classes)
        self.dropout     = nn.Dropout(dropout)

    def forward(self,
                x_raw: Tensor,
                x_freq: Tensor) -> Tensor:
        """
        x_raw:  (batch, 1, n_channels, n_times) 
        x_freq: (batch, n_channels, n_bands)
        """
        # 1) Time‐domain tokens:
        time_tokens = self.patch_embed(x_raw)
        # → (batch, T_patches, emb_size)

        # 2) Frequency‐domain tokens:
        freq_tokens = self.freq_embed(x_freq)
        # → (batch, n_channels, emb_size)

        # 3) Concatenate tokens along seq dimension:
        all_tokens = torch.cat([time_tokens, freq_tokens], dim=1)
        # → (batch, T_patches + n_channels, emb_size)

        # 4) Transformer Encoder:
        z = self.transformer(all_tokens)
        # → (batch, T_patches + n_channels, emb_size)

        # 5) Classification:
        logits = self.classifier(z)  # → (batch, n_classes)
        return logits
