import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value

        self.attention_dim = attention_dim

        self.key_proj = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query_proj = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value_proj = nn.Linear(embedding_dim, attention_dim, bias=False)

        pass

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places

        Q = self.query_proj(embedded)
        K = self.key_proj(embedded)
        V = self.value_proj(embedded)


        scores = (Q @ K.transpose(-2, -1)) / (self.attention_dim ** 0.5)

        lower_triangular = torch.tril(torch.ones(embedded.shape[1], embedded.shape[1]))
        mask = lower_triangular == 0
        scores = scores.masked_fill(mask, float('-inf'))
        scores = nn.functional.softmax(scores, dim=2)

        return torch.round(scores @ V, decimals=4)

