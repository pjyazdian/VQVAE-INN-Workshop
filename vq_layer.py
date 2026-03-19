"""
Vector Quantization (VQ) layers for VQ-VAE implementation.
Adapted from models/quantize_cnn.py to work with motion model architecture.

Key differences from original:
- Original expects (N, width, T) format and does preprocessing/postprocessing
- This version works with (batch, latent_dim) format directly from encoder
- All quantizer types are supported: ema_reset, orig, ema, reset
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizeEMAReset(nn.Module):
    """
    EMA with reset mechanism quantizer.
    Adapted from models/quantize_cnn.py for 2D input (batch, latent_dim).
    Matches T2M-GPT implementation exactly.
    """
    
    def __init__(self, nb_code, code_dim, mu=0.99):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.reset_codebook()
        # Accumulate samples for better initialization (for sequence-level quantization)
        self._init_samples = []
        self._init_batches_needed = max(1, (nb_code // 32) + 1)  # Need enough samples
        
    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim))
        self._init_samples = []

    def _tile(self, x):
        """Tile and add noise to x to get enough samples for codebook initialization."""
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        """
        Initialize codebook from encoder outputs.
        For sequence-level quantization, we accumulate multiple batches for better diversity.
        This matches T2M-GPT's initialization strategy but adapted for sequence-level inputs.
        """
        x_flat = x.view(-1, self.code_dim)
        
        # Accumulate samples across batches for better initialization
        # This is critical for sequence-level quantization where we have limited samples per batch
        self._init_samples.append(x_flat.detach().clone())
        
        if len(self._init_samples) >= self._init_batches_needed:
            # Have enough samples, concatenate and use
            all_samples = torch.cat(self._init_samples, dim=0)
            out = self._tile(all_samples)
            # Clear samples after use
            self._init_samples = []
        else:
            # Not enough samples yet, use current batch with tiling
            # Will be re-initialized when we have enough samples
            out = self._tile(x_flat)
        
        # Initialize codebook (matches T2M-GPT: out[:self.nb_code])
        self.codebook = out[:self.nb_code].clone()
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        code_idx_flat = code_idx.view(-1)
        code_onehot = torch.zeros(self.nb_code, code_idx_flat.shape[0], device=code_idx.device)
        code_onehot.scatter_(0, code_idx_flat.view(1, -1), 1)

        code_count = code_onehot.sum(dim=-1)
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        """
        Update codebook using EMA with reset mechanism.
        Matches T2M-GPT implementation exactly.
        """
        x_flat = x.view(-1, self.code_dim)
        code_idx_flat = code_idx.view(-1)
        
        # Create one-hot encoding: (nb_code, batch_size)
        code_onehot = torch.zeros(self.nb_code, x_flat.shape[0], device=x.device)
        code_onehot.scatter_(0, code_idx_flat.view(1, -1), 1)

        # Compute code sums and counts for this batch
        code_sum = torch.matmul(code_onehot, x_flat)  # (nb_code, code_dim)
        code_count = code_onehot.sum(dim=-1)  # (nb_code,)

        # Prepare random codes for unused entries (reset mechanism)
        out = self._tile(x_flat)
        code_rand = out[:self.nb_code]

        # EMA update: accumulate code sums and counts
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count

        # Reset unused codes: if count < 1.0, use random code instead
        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / (self.code_count.view(self.nb_code, 1) )

        # Update codebook: used codes get EMA update, unused codes get reset to random
        self.codebook = usage * code_update + (1 - usage) * code_rand
        
        # Compute perplexity from current batch usage
        prob = code_count / (torch.sum(code_count) )
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        
        return perplexity

    def quantize(self, x):
        x_flat = x.view(-1, self.code_dim)
        k_w = self.codebook.t()
        distance = (torch.sum(x_flat ** 2, dim=-1, keepdim=True) - 
                   2 * torch.matmul(x_flat, k_w) + 
                   torch.sum(k_w ** 2, dim=0, keepdim=True))
        _, code_idx_flat = torch.min(distance, dim=-1)
        return code_idx_flat.view(x.shape[0])

    def dequantize(self, code_idx):
        code_idx_flat = code_idx.view(-1)
        x_q = F.embedding(code_idx_flat, self.codebook)
        return x_q.view(code_idx.shape[0], self.code_dim)
    
    def forward(self, x):
        """
        Forward pass through quantization layer.
        Matches T2M-GPT implementation.
        """
        # Init codebook if not inited (accumulates samples for better initialization)
        if self.training and not self.init:
            self.init_codebook(x)
            # If still not initialized (not enough samples), return dummy values
            if not self.init:
                # Return dummy values until we have enough samples
                batch_size = x.shape[0]
                x_d = x.clone()
                commit_loss = torch.tensor(0.0, device=x.device)
                perplexity = torch.tensor(1.0, device=x.device)  # Dummy perplexity
                return x_d, commit_loss, perplexity

        # Quantize and dequantize
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)
        
        # Loss: commitment loss (matches T2M-GPT)
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough: preserve gradients (straight-through estimator)
        x_d = x + (x_d - x).detach()
        
        return x_d, commit_loss, perplexity


class Quantizer(nn.Module):
    """
    Original quantizer with beta parameter.
    Adapted from models/quantize_cnn.py for 2D input (batch, latent_dim).
    """
    
    def __init__(self, n_e, e_dim, beta=1.0):
        super(Quantizer, self).__init__()
        self.e_dim = e_dim
        self.n_e = n_e
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        z_flat = z.view(-1, self.e_dim)
        assert z_flat.shape[-1] == self.e_dim

        # Calculate distances
        d = (torch.sum(z_flat ** 2, dim=1, keepdim=True) + 
             torch.sum(self.embedding.weight**2, dim=1) - 
             2 * torch.matmul(z_flat, self.embedding.weight.t()))
        
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices)

        # Compute loss for embedding
        loss = (torch.mean((z_q - z_flat.detach())**2) + 
                self.beta * torch.mean((z_q.detach() - z_flat)**2))

        # Preserve gradients
        z_q = z_flat + (z_q - z_flat).detach()
        z_q = z_q.view(z.shape[0], self.e_dim)

        min_encodings = F.one_hot(min_encoding_indices, self.n_e).type(z_flat.dtype)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        
        return z_q, loss, perplexity

    def quantize(self, z):
        z_flat = z.view(-1, self.e_dim)
        assert z_flat.shape[-1] == self.e_dim

        d = (torch.sum(z_flat ** 2, dim=1, keepdim=True) + 
             torch.sum(self.embedding.weight ** 2, dim=1) - 
             2 * torch.matmul(z_flat, self.embedding.weight.t()))
        min_encoding_indices = torch.argmin(d, dim=1)
        return min_encoding_indices.view(z.shape[0])

    def dequantize(self, indices):
        indices_flat = indices.view(-1)
        z_q = self.embedding(indices_flat)
        return z_q.view(indices.shape[0], self.e_dim)


class QuantizeReset(nn.Module):
    """
    Reset mechanism quantizer without EMA.
    Adapted from models/quantize_cnn.py for 2D input (batch, latent_dim).
    """
    
    def __init__(self, nb_code, code_dim):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.reset_codebook()
        self.codebook = nn.Parameter(torch.randn(nb_code, code_dim))
        
    def reset_codebook(self):
        self.init = False
        self.code_count = None

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        x_flat = x.view(-1, self.code_dim)
        out = self._tile(x_flat)
        self.codebook = nn.Parameter(out[:self.nb_code])
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        code_idx_flat = code_idx.view(-1)
        code_onehot = torch.zeros(self.nb_code, code_idx_flat.shape[0], device=code_idx.device)
        code_onehot.scatter_(0, code_idx_flat.view(1, -1), 1)

        code_count = code_onehot.sum(dim=-1)
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    def update_codebook(self, x, code_idx):
        x_flat = x.view(-1, self.code_dim)
        code_idx_flat = code_idx.view(-1)
        
        code_onehot = torch.zeros(self.nb_code, x_flat.shape[0], device=x.device)
        code_onehot.scatter_(0, code_idx_flat.view(1, -1), 1)

        code_count = code_onehot.sum(dim=-1)

        out = self._tile(x_flat)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_count = code_count
        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()

        self.codebook.data = usage * self.codebook.data + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        
        return perplexity

    def quantize(self, x):
        x_flat = x.view(-1, self.code_dim)
        k_w = self.codebook.t()
        distance = (torch.sum(x_flat ** 2, dim=-1, keepdim=True) - 
                   2 * torch.matmul(x_flat, k_w) + 
                   torch.sum(k_w ** 2, dim=0, keepdim=True))
        _, code_idx_flat = torch.min(distance, dim=-1)
        return code_idx_flat.view(x.shape[0])

    def dequantize(self, code_idx):
        code_idx_flat = code_idx.view(-1)
        x_q = F.embedding(code_idx_flat, self.codebook)
        return x_q.view(code_idx.shape[0], self.code_dim)
    
    def forward(self, x):
        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)
        
        # Quantize and dequantize
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)
        
        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)
        
        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()
        
        return x_d, commit_loss, perplexity


class QuantizeEMA(nn.Module):
    """
    EMA quantizer without reset mechanism.
    Adapted from models/quantize_cnn.py for 2D input (batch, latent_dim).
    """
    
    def __init__(self, nb_code, code_dim, mu=0.99):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.reset_codebook()
        
    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim))

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        x_flat = x.view(-1, self.code_dim)
        out = self._tile(x_flat)
        self.codebook = out[:self.nb_code].clone()
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        code_idx_flat = code_idx.view(-1)
        code_onehot = torch.zeros(self.nb_code, code_idx_flat.shape[0], device=code_idx.device)
        code_onehot.scatter_(0, code_idx_flat.view(1, -1), 1)

        code_count = code_onehot.sum(dim=-1)
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        x_flat = x.view(-1, self.code_dim)
        code_idx_flat = code_idx.view(-1)
        
        code_onehot = torch.zeros(self.nb_code, x_flat.shape[0], device=x.device)
        code_onehot.scatter_(0, code_idx_flat.view(1, -1), 1)

        code_sum = torch.matmul(code_onehot, x_flat)
        code_count = code_onehot.sum(dim=-1)

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count

        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

        self.codebook = code_update
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        
        return perplexity

    def quantize(self, x):
        x_flat = x.view(-1, self.code_dim)
        k_w = self.codebook.t()
        distance = (torch.sum(x_flat ** 2, dim=-1, keepdim=True) - 
                   2 * torch.matmul(x_flat, k_w) + 
                   torch.sum(k_w ** 2, dim=0, keepdim=True))
        _, code_idx_flat = torch.min(distance, dim=-1)
        return code_idx_flat.view(x.shape[0])

    def dequantize(self, code_idx):
        code_idx_flat = code_idx.view(-1)
        x_q = F.embedding(code_idx_flat, self.codebook)
        return x_q.view(code_idx.shape[0], self.code_dim)
    
    def forward(self, x):
        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)

        # Quantize and dequantize
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)
        
        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()
        
        return x_d, commit_loss, perplexity


class VQVAE(nn.Module):
    """
    Main VQ-VAE class that selects quantizer type.
    Similar to VQVAE_251 from models/vqvae.py but adapted for motion model.
    
    Supports quantizer types:
    - 'ema_reset': QuantizeEMAReset (default)
    - 'orig': Quantizer (original with beta)
    - 'ema': QuantizeEMA
    - 'reset': QuantizeReset
    """
    
    def __init__(self, nb_code=512, code_dim=512, quantizer='ema_reset', 
                 mu=0.99, beta=1.0):
        """
        Initialize VQ-VAE quantization layer.
        
        Args:
            nb_code: Number of codebook entries (default: 512)
            code_dim: Dimension of codebook embeddings (should match latent_dim)
            quantizer: Type of quantizer ('ema_reset', 'orig', 'ema', 'reset')
            mu: EMA decay rate for codebook updates (default: 0.99, used for ema_reset and ema)
            beta: Beta parameter for original quantizer (default: 1.0, used for 'orig')
        """
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.quantizer_type = quantizer
        
        # Select quantizer based on type
        if quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(nb_code, code_dim, mu=mu)
        elif quantizer == "orig":
            self.quantizer = Quantizer(nb_code, code_dim, beta=beta)
        elif quantizer == "ema":
            self.quantizer = QuantizeEMA(nb_code, code_dim, mu=mu)
        elif quantizer == "reset":
            self.quantizer = QuantizeReset(nb_code, code_dim)
        else:
            raise ValueError(f"Unknown quantizer type: {quantizer}. Must be one of: 'ema_reset', 'orig', 'ema', 'reset'")
    
    def quantize(self, x):
        """Quantize input to nearest codebook entry."""
        return self.quantizer.quantize(x)
    
    def dequantize(self, code_idx):
        """Dequantize code indices to codebook embeddings."""
        return self.quantizer.dequantize(code_idx)
    
    def forward(self, x):
        """
        Forward pass through quantization layer.
        
        Args:
            x: Input tensor of shape (batch, code_dim)
            
        Returns:
            x_q: Quantized output of shape (batch, code_dim)
            commit_loss: Commitment loss (MSE between input and quantized)
            perplexity: Perplexity of codebook usage
            code_idx: Code indices of shape (batch,)
        """
        # Forward through selected quantizer
        x_q, commit_loss, perplexity = self.quantizer(x)
        
        # Get code indices for return
        code_idx = self.quantize(x)
        
        return x_q, commit_loss, perplexity, code_idx