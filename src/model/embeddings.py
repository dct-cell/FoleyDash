import numpy as np
import torch
import torch.nn as nn

# https://github.com/facebookresearch/DiT


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, dim, frequency_embedding_size, max_period):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.dim = dim
        self.max_period = max_period
        assert dim % 2 == 0, "dim must be even."

        with torch.autocast("cuda", enabled=False):
            freq_indices = torch.arange(
                0, frequency_embedding_size, 2, dtype=torch.float32
            )
            inv_freq = torch.exp(
                -np.log(10000) * freq_indices / frequency_embedding_size
            )
            self.freqs = nn.Buffer(inv_freq, persistent=False)
            freq_scale = 10000 / max_period
            self.freqs *= freq_scale

    def timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py

        args = t[:, None].float() * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb
