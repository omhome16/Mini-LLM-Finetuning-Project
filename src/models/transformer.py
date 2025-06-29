import torch.nn as nn

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

