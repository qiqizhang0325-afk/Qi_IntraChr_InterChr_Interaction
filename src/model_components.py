"""Model components: BiMambaBlock and PositionalEncoding."""

import math

import torch
import torch.nn as nn


class BiMambaBlock(nn.Module):
    """
    BiMamba Simplified Version: Linear complexity bidirectional state space model
    Suitable for ultra-long sequences (millions of markers), complexity O(n)
    instead of O(n²)
    
    Design Philosophy:
    - Use State Space Model to capture long-range dependencies
    - Bidirectional processing: forward and backward scanning, capture global context
    - Linear complexity: suitable for large-scale data
    """
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super(BiMambaBlock, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Forward scan state space parameters
        self.forward_proj = nn.Linear(d_model, d_model * 2)
        self.forward_state = nn.Parameter(torch.randn(d_state, d_model))
        self.forward_gate = nn.Linear(d_model, d_state)
        
        # Backward scan state space parameters
        self.backward_proj = nn.Linear(d_model, d_model * 2)
        self.backward_state = nn.Parameter(torch.randn(d_state, d_model))
        self.backward_gate = nn.Linear(d_model, d_state)
        
        # Output projection
        self.output_proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        
        # Forward scan: accumulate state from left to right
        forward_output = self._scan_forward(x)
        
        # Backward scan: accumulate state from right to left
        backward_output = self._scan_backward(x)
        
        # Fuse bidirectional features
        combined = torch.cat([forward_output, backward_output], dim=-1)
        output = self.output_proj(combined)
        output = self.norm(output)
        output = self.dropout(output)
        
        return output
    
    def _scan_forward(self, x):
        """Forward scan: linear complexity state accumulation (optimized version)"""
        batch, seq_len, d_model = x.shape
        
        # Projection
        proj = self.forward_proj(x)  # (batch, seq_len, d_model*2)
        gate, value = proj.chunk(2, dim=-1)  # Each (batch, seq_len, d_model)
        gate = torch.sigmoid(gate)  # Gating signal
        
        # Calculate gating weights
        gate_weights = torch.sigmoid(self.forward_gate(value))  # (batch, seq_len, d_state)
        
        # State accumulation (optimized: use vectorized operations)
        # Initialize state
        state = torch.zeros(batch, self.d_state, d_model, device=x.device)
        outputs = []
        
        # Accumulate state step by step (linear complexity)
        for i in range(seq_len):
            # Update state: retain old state + fuse new information
            # State space model: state[t] = gate * state[t-1] + (1-gate) * input[t]
            gate_i = gate_weights[:, i].unsqueeze(-1)  # (batch, d_state, 1)
            value_i = value[:, i].unsqueeze(1)  # (batch, 1, d_model)
            
            # State update: weighted combination
            state_update = self.forward_state.unsqueeze(0) * value_i  # (batch, d_state, d_model)
            state = state * gate_i + state_update * (1 - gate_i)
            
            # Output: weighted sum of states
            output_i = (state * gate_i).sum(dim=1)  # (batch, d_model)
            outputs.append(output_i)
        
        return torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
    
    def _scan_backward(self, x):
        """Backward scan: from right to left (optimized version)"""
        batch, seq_len, d_model = x.shape
        
        # Reverse input
        x_rev = torch.flip(x, dims=[1])
        
        # Projection
        proj = self.backward_proj(x_rev)
        gate, value = proj.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        
        # Calculate gating weights
        gate_weights = torch.sigmoid(self.backward_gate(value))  # (batch, seq_len, d_state)
        
        # State accumulation
        state = torch.zeros(batch, self.d_state, d_model, device=x.device)
        outputs = []
        
        for i in range(seq_len):
            gate_i = gate_weights[:, i].unsqueeze(-1)  # (batch, d_state, 1)
            value_i = value[:, i].unsqueeze(1)  # (batch, 1, d_model)
            
            # State update
            state_update = self.backward_state.unsqueeze(0) * value_i
            state = state * gate_i + state_update * (1 - gate_i)
            
            # Output
            output_i = (state * gate_i).sum(dim=1)
            outputs.append(output_i)
        
        # Reverse again to restore original order
        outputs_rev = torch.stack(outputs, dim=1)
        return torch.flip(outputs_rev, dims=[1])


class PositionalEncoding(nn.Module):
    """Positional encoding (for Transformer, capture SNP position information)"""
    def __init__(self, d_model, dropout=0.1, max_len=100000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len × 1 × d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        x: (batch × seq_len × d_model)
        """
        x = x + self.pe[:x.size(1), :].transpose(0, 1)  # Add positional encoding
        return self.dropout(x)



