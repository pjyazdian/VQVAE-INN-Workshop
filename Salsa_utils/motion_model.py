"""
General motion representation model supporting multiple encoder/decoder architectures.
Can work with or without VAE (reparameterization) or with VQ-VAE (vector quantization).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encdec_gru import GRUEncoder, GRUDecoder
from vq_layer import VQVAE


class MotionModel(nn.Module):
    """
    Motion representation model (GRU-only).
    Can work with vanilla AE, VAE, or VQ-VAE.
    """
    
    def __init__(self, input_dim=263, hidden_dim=512, num_layers=2, latent_dim=512, 
                 seq_len=20, dropout=0.1, encoder_type='gru', decoder_type='gru',
                 # Kept for backward compatibility (not used in GRU-only mode)
                 num_heads=8, ff_size=2048, activation='gelu', use_vae=False,
                 # VQ-VAE parameters
                 use_vqvae=False, nb_code=512, quantizer='ema_reset', vq_mu=0.99, vq_beta=1.0):
        """
        Args:
            input_dim: Input feature dimension (263 for HumanML3D)
            hidden_dim: Hidden dimension for GRU encoder/decoder
            num_layers: Number of GRU layers
            latent_dim: Dimension of latent representation
            seq_len: Sequence length (20 frames)
            dropout: Dropout rate
            encoder_type: must be 'gru' (other values are not supported here)
            decoder_type: must be 'gru' (other values are not supported here)
            use_vae: If True, use VAE with reparameterization; if False, use vanilla autoencoder
            use_vqvae: If True, use VQ-VAE with vector quantization (overrides use_vae)
            nb_code: Number of codebook entries for VQ-VAE (default: 512)
            quantizer: Type of quantizer ('ema_reset', 'orig', 'ema', 'reset', default: 'ema_reset')
            vq_mu: EMA decay rate for VQ-VAE codebook updates (default: 0.99, used for ema_reset and ema)
            vq_beta: Beta parameter for original quantizer (default: 1.0, used for 'orig')
        """
        super(MotionModel, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.use_vae = use_vae
        self.use_vqvae = use_vqvae
        
        # VQ-VAE takes precedence over VAE
        if use_vqvae:
            self.use_vae = False
        
        if encoder_type != 'gru' or decoder_type != 'gru':
            raise ValueError("This local MotionModel is GRU-only. Use encoder_type='gru' and decoder_type='gru'.")

        # Create encoder (GRU-only)
        self.encoder = GRUEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=latent_dim,  # Encoder outputs to latent_dim
            dropout=dropout
        )
        
        # VAE-specific components: mean and logvar projections
        if use_vae and not use_vqvae:
            # Project encoder output to mean and logvar
            self.fc_mean = nn.Linear(latent_dim, latent_dim)
            self.fc_logvar = nn.Linear(latent_dim, latent_dim)
            
            # Initialize with smaller weights to prevent large initial logvar
            nn.init.xavier_uniform_(self.fc_mean.weight, gain=0.1)
            nn.init.xavier_uniform_(self.fc_logvar.weight, gain=0.01)
            nn.init.constant_(self.fc_logvar.bias, -2.0)  # Start with small variance
        
        # VQ-VAE-specific components: vector quantization layer
        if use_vqvae:
            # VQ layer quantizes encoder output
            self.vq_layer = VQVAE(
                nb_code=nb_code,
                code_dim=latent_dim,  # Codebook dimension matches latent_dim
                quantizer=quantizer,
                mu=vq_mu,
                beta=vq_beta
            )
        
        # Create decoder (GRU-only)
        self.decoder = GRUDecoder(
            input_dim=latent_dim,  # Decoder takes latent_dim as input
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=input_dim,
            seq_len=seq_len,
            dropout=dropout
        )
    
    def reparameterize(self, mean, logvar):
        """Reparameterization trick for VAE with numerical stability."""
        # Clamp logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def encode(self, x):
        """
        Encode input to latent representation.
        
        Returns:
            For VQ-VAE: (z, commit_loss, perplexity, code_idx)
            For VAE: (z, mean, logvar)
            For vanilla AE: (z, mean, logvar) where mean=encoded, logvar=zeros
        """
        # Encoder outputs clean representation
        encoded = self.encoder(x)  # (batch, latent_dim)
        
        if self.use_vqvae:
            # Vector quantization: quantize encoder output directly (matches T2M-GPT)
            z, commit_loss, perplexity, code_idx = self.vq_layer(encoded)
            return z, commit_loss, perplexity, code_idx
        elif self.use_vae:
            # Project to mean and logvar, then reparameterize
            mean = self.fc_mean(encoded)
            logvar = self.fc_logvar(encoded)
            logvar = torch.clamp(logvar, min=-10, max=10)
            z = self.reparameterize(mean, logvar)
            return z, mean, logvar
        else:
            # Vanilla autoencoder: use encoder output directly
            mean = encoded
            logvar = torch.zeros_like(encoded)  # Dummy logvar for compatibility
            z = encoded
            return z, mean, logvar
    
    def decode(self, z, first_frame=None):
        """Decode latent representation to motion."""
        return self.decoder(z, first_frame)
    
    def forward(self, x):
        """
        Forward pass through model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len=20, input_dim=263)
        
        Returns:
            For VQ-VAE: (recon_x, z, commit_loss, perplexity, code_idx)
            For VAE: (recon_x, mean, logvar, z)
            For vanilla AE: (recon_x, mean, logvar, z)
        """
        # Encode
        encode_result = self.encode(x)
        
        if self.use_vqvae:
            z, commit_loss, perplexity, code_idx = encode_result
        else:
            z, mean, logvar = encode_result
        
        # Use first frame of input for decoder
        first_frame = x[:, 0, :]  # (batch, input_dim)
        recon_x = self.decoder(z, first_frame)
        
        if self.use_vqvae:
            return recon_x, z, commit_loss, perplexity, code_idx
        else:
            return recon_x, mean, logvar, z
    
    def sample(self, batch_size, device, first_frame=None):
        """Sample from prior distribution (only for VAE)."""
        if not self.use_vae:
            raise ValueError("sample() is only available when use_vae=True")
        z = torch.randn(batch_size, self.latent_dim, device=device)
        return self.decoder(z, first_frame)
    
    def inference_encode(self, x):
        """
        Inference encoding: encode batch of windows and return latents/tokens.
        Mirrors the encoding part of forward().
        
        Args:
            x: Input tensor of shape (batch_size, seq_len=20, input_dim=263)
        
        Returns:
            For VQ-VAE: (z, code_idx) where z is quantized latents, code_idx is token indices
            For VAE/Vanilla: (z, mean, logvar) where z is latent representation
        """
        encode_result = self.encode(x)
        
        if self.use_vqvae:
            z, commit_loss, perplexity, code_idx = encode_result
            return z, code_idx
        else:
            z, mean, logvar = encode_result
            return z, mean, logvar
    
    def inference_decode_autoregressive(self, latents_or_tokens, first_frame, num_windows):
        """
        Autoregressive decoding for long sequences.
        Mirrors the decoding part of forward() but applied autoregressively.
        
        Args:
            latents_or_tokens: For VQ-VAE: tuple (z, code_idx) where z is (num_windows, latent_dim)
                              For VAE/Vanilla: tuple (z, mean, logvar) where z is (num_windows, latent_dim)
            first_frame: First frame of the sequence (1, input_dim) - ground truth
            num_windows: Number of windows to generate
        
        Returns:
            Generated motion sequence of shape (num_windows * seq_len, input_dim)
        """
        self.eval()
        
        with torch.no_grad():
            # Extract latents based on model type
            if self.use_vqvae:
                z, code_idx = latents_or_tokens
                # z is already quantized from VQ layer
            else:
                z, mean, logvar = latents_or_tokens
                # z is already the latent representation (reparameterized for VAE or direct for vanilla)
            
            # Ensure z has correct shape: (num_windows, latent_dim)
            if z.dim() == 1:
                z = z.unsqueeze(0)  # (1, latent_dim) -> (num_windows=1, latent_dim)
            elif z.dim() == 2 and z.shape[0] != num_windows:
                # If we have batch of windows, ensure it matches num_windows
                if z.shape[0] == 1:
                    z = z.repeat(num_windows, 1)  # Repeat for all windows
            
            # Initialize with first frame (ground truth)
            current_first_frame = first_frame  # (1, input_dim)
            
            # Store all generated frames
            all_generated_frames = []
            
            # Generate each window autoregressively
            for window_idx in range(num_windows):
                # Get latent for this window
                z_window = z[window_idx:window_idx+1]  # (1, latent_dim)
                
                # Decode this window using current first frame
                generated_window = self.decoder(z_window, current_first_frame)  # (1, seq_len, input_dim)
                
                # Extract frames from generated window
                generated_frames = generated_window[0]  # (seq_len, input_dim)
                all_generated_frames.append(generated_frames)
                
                # Use last frame of this window as first frame for next window
                # generated_frames[-1:] gives (1, input_dim) which is (batch=1, input_dim) - correct shape
                current_first_frame = generated_frames[-1:]  # (1, input_dim)
            
            # Concatenate all windows
            full_sequence = torch.cat(all_generated_frames, dim=0)  # (num_windows * seq_len, input_dim)
            
            return full_sequence


def vae_loss(recon_x, x, recon_weight=1.0, kl_weight=0.0001, use_vae=True,
             use_vqvae=False, mean=None, logvar=None, commit_loss=None, commit_weight=0.02,
             loss_vel_weight=0.0):
    """
    Loss function for VAE, VQ-VAE, or vanilla autoencoder.
    Matches T2M-GPT loss formulation: reconstruction + commitment + velocity.
    
    Args:
        recon_x: Reconstructed motion (batch, seq_len, dim)
        x: Original motion (batch, seq_len, dim)
        recon_weight: Weight for reconstruction loss
        kl_weight: Weight for KL divergence loss (VAE only)
        use_vae: If True, include KL loss (VAE mode)
        use_vqvae: If True, include commitment loss (VQ-VAE mode)
        mean: Mean of latent distribution (required if use_vae=True)
        logvar: Log variance of latent distribution (required if use_vae=True)
        commit_loss: Commitment loss from VQ-VAE (required if use_vqvae=True)
        commit_weight: Weight for commitment loss (VQ-VAE only, default: 0.02)
        loss_vel_weight: Weight for velocity loss (default: 0.0, set to 0.1 for VQ-VAE)
    
    Returns:
        total_loss: Total loss
        recon_loss: Reconstruction loss (MSE)
        kl_loss: KL divergence loss (0.0 if not VAE) or commitment loss (if VQ-VAE)
        vel_loss: Velocity loss (0.0 if loss_vel_weight=0.0)
    """
    # Reconstruction loss (L1) - matches T2M-GPT default (can be L1, L2, or SmoothL1)
    # T2M-GPT uses L1 loss by default, which is more robust
    recon_loss = F.l1_loss(recon_x, x, reduction='mean')
    
    # Check for NaN in reconstruction
    if torch.isnan(recon_loss):
        print("Warning: NaN in reconstruction loss!")
        recon_loss = torch.tensor(0.0, device=recon_loss.device)
    
    # Velocity loss: frame-to-frame differences (temporal smoothness)
    # Computes L1 loss between consecutive frame differences (matches T2M-GPT)
    # T2M-GPT uses the same loss type (L1) for both reconstruction and velocity
    vel_loss = torch.tensor(0.0, device=recon_loss.device)
    if loss_vel_weight > 0.0 and recon_x.shape[1] > 1:  # Need at least 2 frames
        # Compute velocity: diff[i] = frame[i+1] - frame[i]
        pred_vel = recon_x[:, 1:] - recon_x[:, :-1]  # (batch, seq_len-1, dim)
        gt_vel = x[:, 1:] - x[:, :-1]  # (batch, seq_len-1, dim)
        vel_loss = F.l1_loss(pred_vel, gt_vel, reduction='mean')
        
        # Check for NaN in velocity loss
        if torch.isnan(vel_loss) or torch.isinf(vel_loss):
            print("Warning: NaN/Inf in velocity loss! Setting to 0.")
            vel_loss = torch.tensor(0.0, device=recon_loss.device)
    
    # VQ-VAE loss: reconstruction + commitment + velocity
    if use_vqvae:
        if commit_loss is None:
            raise ValueError("commit_loss must be provided when use_vqvae=True")
        
        # Commitment loss (already computed in VQ layer)
        commit_loss_val = commit_loss
        
        # Check for NaN in commitment loss
        if torch.isnan(commit_loss_val) or torch.isinf(commit_loss_val):
            print("Warning: NaN/Inf in commitment loss! Clamping...")
            commit_loss_val = torch.clamp(commit_loss_val, min=-1e6, max=1e6)
            if torch.isnan(commit_loss_val):
                commit_loss_val = torch.tensor(0.0, device=recon_loss.device)
        
        # Total loss: reconstruction + commitment + velocity (matches T2M-GPT)
        total_loss = recon_weight * recon_loss + commit_weight * commit_loss_val + loss_vel_weight * vel_loss
        
        # Final NaN check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: NaN/Inf in total loss! recon={recon_loss:.4f}, commit={commit_loss_val:.4f}, vel={vel_loss:.4f}")
            total_loss = torch.clamp(total_loss, min=-1e6, max=1e6)
            if torch.isnan(total_loss):
                total_loss = recon_weight * recon_loss  # Fallback to reconstruction only
        
        return total_loss, recon_loss, commit_loss_val, vel_loss
    
    # VAE loss: reconstruction + KL divergence
    elif use_vae:
        if mean is None or logvar is None:
            raise ValueError("mean and logvar must be provided when use_vae=True")
        
        # Clamp logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        # KL divergence loss with numerical stability
        # Use more stable formula: -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
        # Clamp exp(logvar) to prevent overflow
        var = torch.clamp(torch.exp(logvar), min=1e-8, max=1e8)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - var, dim=1)
        kl_loss = torch.mean(kl_loss)
        
        # Check for NaN in KL loss
        if torch.isnan(kl_loss) or torch.isinf(kl_loss):
            print("Warning: NaN/Inf in KL loss! Clamping...")
            kl_loss = torch.clamp(kl_loss, min=-1e6, max=1e6)
            if torch.isnan(kl_loss):
                kl_loss = torch.tensor(0.0, device=kl_loss.device)
        
        # Total loss: reconstruction + KL + velocity (if enabled)
        total_loss = recon_weight * recon_loss + kl_weight * kl_loss + loss_vel_weight * vel_loss
        
        # Final NaN check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: NaN/Inf in total loss! recon={recon_loss:.4f}, kl={kl_loss:.4f}, vel={vel_loss:.4f}")
            total_loss = torch.clamp(total_loss, min=-1e6, max=1e6)
            if torch.isnan(total_loss):
                total_loss = recon_weight * recon_loss  # Fallback to reconstruction only
        
        return total_loss, recon_loss, kl_loss, vel_loss
    
    else:
        # Vanilla autoencoder: reconstruction + velocity (if enabled)
        kl_loss = torch.tensor(0.0, device=recon_loss.device)
        total_loss = recon_weight * recon_loss + loss_vel_weight * vel_loss
        
        # Final NaN check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: NaN/Inf in total loss! recon={recon_loss:.4f}, vel={vel_loss:.4f}")
            total_loss = torch.clamp(total_loss, min=-1e6, max=1e6)
            if torch.isnan(total_loss):
                total_loss = recon_weight * recon_loss
        
        return total_loss, recon_loss, kl_loss, vel_loss

