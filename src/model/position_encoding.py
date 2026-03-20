import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_seq_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, hidden_size)
        
        # Create position tensor
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Compute div_term for sinusoidal functions
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * 
                           (-math.log(10000.0) / hidden_size))
        
        # Compute positional encodings
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter but will be saved with the model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input tensor
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            x: Tensor with positional encoding added (same shape)
        """
        # Add positional encoding
        x = x + self.pe[:, :x.size(1)]
        
        # Apply dropout
        x = self.dropout(x)
        
        return x