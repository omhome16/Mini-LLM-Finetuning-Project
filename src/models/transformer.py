import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
import math

@dataclass
class ModelArgs:
    """
    Configuration of Pythia-70M.
    """
    dim: int = 512  # Embedding dimension
    n_layers: int = 6  # Number of decoder blocks
    n_heads: int = 8  # Number of attention heads
    vocab_size: int = 50304  # Pythia-70m vocabulary size
    norm_eps: float = 1e-5  # Epsilon for RMSNorm
    rope_theta: float = 10000.0  # Theta for RoPE

    max_batch_size: int = 4  # Max batch size
    max_seq_len: int = 512  # Max sequence length

    epochs: int = 3  # Total training epochs
    log_interval: int = 10  # Interval to print logs
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6):

        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        rms = x.norm(2, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.weight * (x / (rms + self.eps))


class RoPE(nn.Module):

    def __init__(self, head_dim: int, rope_theta: float = 10000.0):

        super().__init__()
        self.head_dim = head_dim

        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:

        x1, x2 = x[..., : self.head_dim // 2], x[..., self.head_dim // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:

        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim/2)

        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(1).repeat(1, x.size(1), 1, 1)  # (B, H, S, D_h)
        sin = emb.sin().unsqueeze(1).repeat(1, x.size(1), 1, 1)  # (B, H, S, D_h)

        x_rotated = self.rotate_half(x)
        return (x * cos) + (x_rotated * sin)


class MultiHeadAttention(nn.Module):

    def __init__(self, config: ModelArgs):

        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = self.dim // self.n_heads

        self.q_proj = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        self.rope = RoPE(self.head_dim, rope_theta=config.rope_theta)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        batch_size, seq_len, _ = hidden_states.shape

        queries = self.q_proj(hidden_states).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, S, D_h)
        keys = self.k_proj(hidden_states).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, S, D_h)
        values = self.v_proj(hidden_states).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, S, D_h)

        queries = self.rope(queries, seq_len)
        keys = self.rope(keys, seq_len)

        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device), diagonal=1)
        scores.masked_fill_(causal_mask, float('-inf'))

        if attention_mask is not None:

            attention_mask = attention_mask[:, None, None, :].masked_fill(attention_mask[:, None, None, :]==0, float('-inf'))
            scores += attention_mask

        attention_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        output = torch.matmul(attention_weights, values)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        output = self.o_proj(output)

        return output


class FeedForward(nn.Module):


    def __init__(self, config: ModelArgs):

        super().__init__()
        self.dim = config.dim
        # Pythia FFN is 4x the hidden dim
        self.hidden_dim = self.dim * 4

        self.fc1 = nn.Linear(self.dim, self.hidden_dim, bias=True)
        self.fc2 = nn.Linear(self.hidden_dim, self.dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.fc2(F.gelu(self.fc1(x)))


class DecoderBlock(nn.Module):

    def __init__(self, config: ModelArgs):

        super().__init__()
        self.dim = config.dim

        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attention = MultiHeadAttention(config)

        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.feed_forward = FeedForward(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        norm_hidden_states = self.attention_norm(hidden_states)
        attn_output = self.attention(norm_hidden_states, attention_mask=attention_mask)
        hidden_states = hidden_states + attn_output  # Residual connection

        norm_hidden_states = self.ffn_norm(hidden_states)
        ffn_output = self.feed_forward(norm_hidden_states)
        hidden_states = hidden_states + ffn_output  # Residual connection

        return hidden_states


class CausalLM(nn.Module):

    def __init__(self, config: ModelArgs):

        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.dim)

        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.n_layers)]
        )

        self.final_norm = RMSNorm(config.dim, eps=config.norm_eps)

        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embedding.weight

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        hidden_states = self.embedding(input_ids)

        for layer in self.decoder_blocks:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return logits, loss