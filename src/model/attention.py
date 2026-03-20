import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Ensure hidden_size is divisible by num_heads
        assert self.head_dim * num_heads == self.hidden_size, "hidden_size must be divisible by num_heads"
        
        # Linear layers for Q, K, V
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        
        # Output linear layer
        self.output_linear = nn.Linear(hidden_size, hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor for attention
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query, key, value, attention_mask=None):
        """
        Forward pass for multi-head attention
        
        Args:
            query: Tensor of shape (batch_size, seq_len, hidden_size)
            key: Tensor of shape (batch_size, seq_len, hidden_size)
            value: Tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional tensor of shape (batch_size, seq_len) or (batch_size, 1, seq_len)
            
        Returns:
            attention_output: Tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # Split into heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to match attention scores shape
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            
            # Set masked positions to large negative value
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        # Apply softmax
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_probs, V)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.hidden_size)
        
        # Final linear projection
        attention_output = self.output_linear(attention_output)
        
        return attention_output