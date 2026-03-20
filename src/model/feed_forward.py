import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super(FeedForward, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # First linear layer (expand)
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        
        # Second linear layer (reduce)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        self.activation = F.gelu
    
    def forward(self, x):
        """
        Forward pass through feed-forward network
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            x: Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # First linear transformation with activation
        x = self.activation(self.dense1(x))
        
        # Apply dropout
        x = self.dropout(x)
        
        # Second linear transformation
        x = self.dense2(x)
        
        return x