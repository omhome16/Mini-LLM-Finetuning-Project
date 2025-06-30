# decoder block

import torch.nn as nn

"""
llama 3.1 8b config

    vocab_size: int = 128256
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # Grouped-Query Attention (GQA)
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    max_position_embeddings: int = 131072
    pad_token_id: int = 128255 # Corresponds to <|eot_id|>
    bos_token_id: int = 128000
    eos_token_id: int = 128001
    
"""

class RMSNorm(nn.Module):






class MultiHeadAttention (nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads,  rope_theta=10000):

        super().__init__()

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.kv_head_dim = hidden_size // num_kv_heads
        self.num_query_groups = num_heads // num_kv_heads
        self.rope_theta = rope_theta

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, self.kv_head_dim * num_kv_heads)
        self.v_proj = nn.Linear(hidden_size, self.kv_head_dim * num_kv_heads)
        self.o_proj = nn.Linear(hidden_size, hidden_size)








    def forward(self, x):

