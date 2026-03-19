"""
GRU-based encoder and decoder for motion sequences.
Clean implementation without VAE-specific components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUEncoder(nn.Module):
    """Bidirectional GRU-based encoder for motion sequences with FC layers."""
    
    def __init__(self, input_dim=263, hidden_dim=512, num_layers=2, output_dim=512, dropout=0.1):
        """
        Args:
            input_dim: Input feature dimension (263 for HumanML3D)
            hidden_dim: Hidden dimension of GRU layers
            num_layers: Number of GRU layers
            output_dim: Output dimension (will be used as latent_dim in MotionModel)
            dropout: Dropout rate
        """
        super(GRUEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bidirectional = True
        
        # Input FC layer with activation
        self.fc_in = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional GRU layers
        self.gru = nn.GRU(
            hidden_dim,  # Input after FC
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Bidirectional!
        )
        
        # GRU output dimension (bidirectional: 2 * hidden_dim)
        gru_output_dim = hidden_dim * 2
        
        # Intermediate FC layer
        self.fc_intermediate = nn.Sequential(
            nn.Linear(gru_output_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        
        # Output projection (clean, no VAE-specific components)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len=20, input_dim=263)
        
        Returns:
            output: Encoded representation (batch_size, output_dim)
        """
        # Store input for debugging (always, not just in debug mode)
        self._last_input = x.detach().clone()
        
        # DEBUG: Track input to fc_in (only for problematic batches or periodic checks)
        # We'll use a global counter or check if this is a training batch
        debug_mode = getattr(self, '_debug_mode', False)
        if debug_mode:
            print(f"\n[DEBUG ENCODER] Input to encoder:")
            print(f"  x shape: {x.shape}")
            print(f"  x stats: min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")
            print(f"  x has NaN: {torch.isnan(x).any().item()}, has Inf: {torch.isinf(x).any().item()}")
        
        # x: (batch, 20, 263)
        # Apply input FC
        x_fc_in = self.fc_in(x)  # (batch, 20, hidden_dim)
        
        # Store fc_in output for debugging
        self._last_fc_in_output = x_fc_in.detach().clone()
        
        # DEBUG: Track output of fc_in
        if debug_mode:
            print(f"\n[DEBUG ENCODER] Output from fc_in:")
            print(f"  x_fc_in shape: {x_fc_in.shape}")
            print(f"  x_fc_in stats: min={x_fc_in.min().item():.4f}, max={x_fc_in.max().item():.4f}, mean={x_fc_in.mean().item():.4f}, std={x_fc_in.std().item():.4f}")
            print(f"  x_fc_in has NaN: {torch.isnan(x_fc_in).any().item()}, has Inf: {torch.isinf(x_fc_in).any().item()}")
            # Check fc_in weights (fc_in is Sequential with Linear at index 0)
            fc_in_linear = self.fc_in[0]
            if isinstance(fc_in_linear, torch.nn.Linear):
                print(f"  fc_in[0] (Linear) weight stats: min={fc_in_linear.weight.min().item():.4f}, max={fc_in_linear.weight.max().item():.4f}, mean={fc_in_linear.weight.mean().item():.4f}, std={fc_in_linear.weight.std().item():.4f}")
                print(f"  fc_in[0] weight has NaN: {torch.isnan(fc_in_linear.weight).any().item()}, has Inf: {torch.isinf(fc_in_linear.weight).any().item()}")
                print(f"  fc_in[0] bias stats: min={fc_in_linear.bias.min().item():.4f}, max={fc_in_linear.bias.max().item():.4f}, mean={fc_in_linear.bias.mean().item():.4f}")
                print(f"  fc_in[0] bias has NaN: {torch.isnan(fc_in_linear.bias).any().item()}, has Inf: {torch.isinf(fc_in_linear.bias).any().item()}")
        
        x = x_fc_in
        
        # Bidirectional GRU
        gru_out, hidden = self.gru(x)
        # gru_out: (batch, 20, hidden_dim * 2) [bidirectional]
        # hidden: (num_layers * 2, batch, hidden_dim) [bidirectional]
        
        # Concatenate forward and backward hidden states from last layer
        forward_hidden = hidden[self.num_layers - 1]  # (batch, hidden_dim)
        backward_hidden = hidden[2 * self.num_layers - 1]  # (batch, hidden_dim)
        last_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)  # (batch, hidden_dim * 2)
        
        # Apply intermediate FC
        last_hidden = self.fc_intermediate(last_hidden)  # (batch, hidden_dim)
        
        # Project to output dimension
        output = self.fc_out(last_hidden)  # (batch, output_dim)
        
        return output


class GRUDecoder(nn.Module):
    """GRU-based decoder for motion sequences with first frame input and autoregressive reconstruction."""
    
    def __init__(self, input_dim=512, hidden_dim=512, num_layers=2, output_dim=263, seq_len=20, dropout=0.1):
        """
        Args:
            input_dim: Input dimension (latent_dim from MotionModel)
            hidden_dim: Hidden dimension of GRU layers
            num_layers: Number of GRU layers
            output_dim: Output feature dimension (263 for HumanML3D)
            seq_len: Sequence length (20 frames)
            dropout: Dropout rate
        """
        super(GRUDecoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        # Project input to initial hidden state
        self.input_to_hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * num_layers)
        )
        
        # Input FC for first frame
        self.fc_first_frame = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GRU layers
        self.gru = nn.GRU(
            hidden_dim,  # Input after FC
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output projection with activation
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.heavy_dropout = nn.Dropout(0.95)
    def forward(self, z, first_frame=None):
        """
        Args:
            z: Latent tensor of shape (batch_size, input_dim)
            first_frame: First frame of original motion (batch_size, output_dim=263)
                        If None, uses zeros
        
        Returns:
            output: Reconstructed motion (batch_size, seq_len=20, output_dim=263)
        """
        batch_size = z.size(0)
        
        # Project input to initial hidden state
        hidden = self.input_to_hidden(z)  # (batch, hidden_dim * num_layers)
        hidden = hidden.view(batch_size, self.num_layers, self.hidden_dim)
        hidden = hidden.permute(1, 0, 2).contiguous()  # (num_layers, batch, hidden_dim)
        
        # Use first frame as initial input, or zeros if not provided
        if first_frame is None:
            decoder_input = torch.zeros(batch_size, 1, self.output_dim, device=z.device)
        else:
            decoder_input = first_frame.unsqueeze(1)  # (batch, 1, output_dim)
        decoder_input = self.heavy_dropout(decoder_input)
        # Apply FC to first frame input
        decoder_input = self.fc_first_frame(decoder_input)  # (batch, 1, hidden_dim)
        
        # Decode step by step (autoregressive)
        outputs = []
        for t in range(self.seq_len):
            gru_out, hidden = self.gru(decoder_input, hidden)
            # gru_out: (batch, 1, hidden_dim)
            output = self.fc_out(gru_out)  # (batch, 1, output_dim)
            outputs.append(output)
            
            # Use output as next input (autoregressive)
            # Apply FC to prepare for next GRU step
            decoder_input = self.fc_first_frame(output)  # (batch, 1, hidden_dim)
        
        # Concatenate all outputs
        output = torch.cat(outputs, dim=1)  # (batch, seq_len, output_dim)
        
        return output

