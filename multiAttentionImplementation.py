import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Scaled Dot-Product Attention Function
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    # (batch, heads, seq_len, head_dim) @ (batch, heads, head_dim, seq_len) --> (batch, heads, seq_len, seq_len)
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    # (batch, heads, seq_len, seq_len) @ (batch, heads, seq_len, head_dim) --> (batch, heads, seq_len, head_dim)
    values = torch.matmul(attention, v)
    return values, attention

# Multi-Head Attention Class
class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, d_model, num_heads):
        super().__init__()
        self.input_dim = input_dim      # Input embedding size
        self.d_model = d_model          # Model embedding size (output of self-attention)
        self.num_heads = num_heads      # Number of parallel attention heads
        self.head_dim = d_model // num_heads  # Dimensionality per head

        # For efficiency, compute Q, K, V for all heads at once with a single linear layer
        self.qkv_layer = nn.Linear(input_dim, 3 * d_model)
        # Final projection, combines all heads' outputs
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, sequence_length, input_dim = x.size()
        print(f"x.size(): {x.size()}")  # Input shape

        # Step 1: Project x into concatenated q, k, v for ALL heads at once
        qkv = self.qkv_layer(x)
        print(f"qkv.size(): {qkv.size()}")  # Shape: (batch, seq_len, 3 * d_model)

        # Step 2: reshape into (batch, seq_len, num_heads, 3 * head_dim)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        print(f"qkv.size(): {qkv.size()}")

        # Step 3: Rearrange to (batch, num_heads, seq_len, 3 * head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        print(f"qkv.size(): {qkv.size()}")

        # Step 4: Split the last dimension into q, k, v (each get last dimension of head_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # Each: (batch, num_heads, seq_len, head_dim)
        print(f"q size: {q.size()}, k size: {k.size()}, v size: {v.size()}")

        # Step 5: Apply scaled dot product attention to get outputs (contextualized values) and attention weights
        values, attention = scaled_dot_product(q, k, v, mask)
        print(f"values.size(): {values.size()}, attention.size: {attention.size()}")

        # Step 6: Merge the heads (permute before reshape)
        values = values.permute(0, 2, 1, 3)   # (batch, seq_len, heads, head_dim)
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        print(f"values.size(): {values.size()}")

        # Step 7: Final linear projection to match d_model
        out = self.linear_layer(values)
        print(f"out.size(): {out.size()}")
        return out

# Run With Printouts
# Model/inputs setup
input_dim = 1024   # Input feature size per token
d_model = 512      # Embedding/model size (must divide num_heads)
num_heads = 8
batch_size = 30
sequence_length = 5

# Create random input
x = torch.randn((batch_size, sequence_length, input_dim))

# Instantiate MultiheadAttention class and run
model = MultiheadAttention(input_dim, d_model, num_heads)
output = model.forward(x)