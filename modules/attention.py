import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    ### YOUR CODE HERE
    """
    key dims: [bs, num_attention_heads, t, d] (2,12,8,64)
    query dims: [bs, num_attention_heads, t, d] (2,12,8,64)
    value dims: [bs, num_attention_heads, t, d] (2,12,8,64)
    attention_mask dims: [bs, 1, 1, seq_len] (2,1,1,8)
    """
    # print(key.shape)
    # print(query.shape)
    # print(value.shape)
    d_k = query.shape[-1]
    
    # Compute attention scores
    # scores dim: [bs, num_attention_heads, t, sqrt(d)]
    scores = torch.matmul(query, key.transpose(-2, -1))
    # print(scores.shape)
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=scores.dtype, device=scores.device))
    
    # Apply causal mask
    causal_mask = torch.triu(torch.ones(scores.shape, dtype=torch.bool), diagonal=1)
    # print('attention_mask:', attention_mask)
    scores = scores + attention_mask
    
    scores = scores.masked_fill(causal_mask, float('-inf'))
    # Compute attention probabilities
    # print(scores.shape)
    # print(attention_mask.shape)
    attn_probs = torch.nn.functional.softmax(scores, dim=-1)
    attn_probs = self.dropout(attn_probs)
    # print(attn_probs.shape)

    # Compute final attention output
    attn_output = torch.matmul(attn_probs, value)
    
    # Reshape back to [batch_size, seq_len, hidden_state]
    attn_output = rearrange(attn_output, 'b h t d -> b t (h d)')

    # DEBUGGING
    # print("Attention scores (before softmax):", scores[0, 0, :, :])
    # print("Attention probabilities:", attn_probs[0, 0, :, :])
    # print("Attention sum:", attn_probs.sum(dim=-1))  # Should be 1 for each sequence position
    
    return attn_output


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)

    return attn_value
