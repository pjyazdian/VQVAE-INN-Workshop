"""
Minimal utilities for 03_Salsa_Dance notebook.
Uses the Salsa motion representation codebase without modifying it.
"""

import sys
from pathlib import Path

# Add Salsa motion_representation and in2IN to path (no changes to their code)
SALSA_MOTION_REP = Path(
    "/local-scratch/localhome/pjomeyaz/Payam_Files/Projects/Salsa_Dance/"
    "scripts/New_2025/Salsa-Agent/motion_representation"
)
SALSA_AGENT = SALSA_MOTION_REP.parent
IN2IN_PATH = SALSA_AGENT.parent / "Download" / "in2IN"
for p in [str(SALSA_AGENT), str(IN2IN_PATH)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from motion_representation.models import MotionModel
from motion_representation.data.motion_dataset import create_dataloader

import torch
import numpy as np
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Model (vanilla GRU AE - matches Salsa encdec_gru exactly)
# -----------------------------------------------------------------------------

def get_salsa_model(
    input_dim=263,
    hidden_dim=512,
    num_layers=2,
    latent_dim=512,
    seq_len=20,
    dropout=0.1,
):
    """
    Vanilla GRU autoencoder matching Salsa motion_representation architecture.
    Uses MotionModel with use_vae=False, use_vqvae=False.
    """
    return MotionModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        latent_dim=latent_dim,
        seq_len=seq_len,
        dropout=dropout,
        encoder_type="gru",
        decoder_type="gru",
        use_vae=False,
        use_vqvae=False,
    )


# -----------------------------------------------------------------------------
# Data loaders (uses Salsa create_dataloader, no changes to their code)
# -----------------------------------------------------------------------------

def get_salsa_loaders(
    salsa_root: str | Path,
    representation_type: str = "humanml3d",
    window_size: int = 20,
    stride: int = 10,
    batch_size: int = 32,
    num_workers: int = 0,
    use_both_roles: bool = True,
    normalize: bool = True,
):
    """
    Create train and val DataLoaders for Salsa motion data.
    Uses lmdb_train and lmdb_val from salsa_root/dataset_processed_New/lmdb_Salsa_pair/.
    Returns (train_loader, val_loader).

    What we load:
    - representation_type='interhuman': Canonicalized motion from Salsa dance pairs.
      Each sample: (T, 262) where T=19 (window_size-1). Format: first 66 dims = 22 joint
      positions (x,y,z) per frame; remaining 196 = rotations. Data from both leader and
      follower when use_both_roles=True. Uses cache lmdb_train_interhuman_20frames_cache.
    - representation_type='humanml3d': (T, 263) HumanML3D format, T=20.
    """
    salsa_root = Path(salsa_root)
    base = salsa_root / "dataset_processed_New" / "lmdb_Salsa_pair"
    lmdb_train = str(base / "lmdb_train")
    lmdb_val = str(base / "lmdb_val")

    if not Path(lmdb_train).exists():
        raise FileNotFoundError(f"LMDB train not found: {lmdb_train}")

    # Minimal args object (MotionWindowDataset stores it; no CLI parsing)
    class Args:
        pass
    args = Args()
    args.lmdb_dir = lmdb_train
    args.parent_dir = str(salsa_root)

    train_loader = create_dataloader(
        args=args,
        lmdb_dir=lmdb_train,
        window_size=window_size,
        stride=stride,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        use_both_roles=use_both_roles,
        normalize=normalize,
        representation_type=representation_type,
    )

    val_loader = None
    if Path(lmdb_val).exists():
        val_loader = create_dataloader(
            args=args,
            lmdb_dir=lmdb_val,
            window_size=window_size,
            stride=stride,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            use_both_roles=use_both_roles,
            normalize=normalize,
            representation_type=representation_type,
        )
    else:
        # Fallback: use train for both (no val split)
        val_loader = create_dataloader(
            args=args,
            lmdb_dir=lmdb_train,
            window_size=window_size,
            stride=stride,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            use_both_roles=use_both_roles,
            normalize=normalize,
            representation_type=representation_type,
        )

    return train_loader, val_loader


# -----------------------------------------------------------------------------
# Training (matches our AETrainer style)
# -----------------------------------------------------------------------------

class SalsaAETrainer:
    """Vanilla AE trainer for Salsa model. MSE reconstruction loss."""

    def __init__(self, model, train_loader, val_loader, learning_rate=2e-4, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0.0)
        # Loss tracking for visualization
        self.global_step = 0
        self.batch_losses = []  # list of (step, loss)
        self.epoch_train_losses = []  # list of floats
        self.epoch_val_losses = []  # list of floats

    def loss_fn(self, recon, x):
        return torch.nn.functional.mse_loss(recon, x)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in tqdm(self.train_loader, leave=False):
            data = batch.to(self.device)
            self.optimizer.zero_grad()
            recon, mean, logvar, z = self.model(data)
            loss = self.loss_fn(recon, data)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            # Track per-batch loss for plotting
            self.batch_losses.append((self.global_step, loss.item()))
            self.global_step += 1
        mean_loss = total_loss / n_batches
        print(f"====> Epoch: {epoch} Train loss: {mean_loss:.4f}")
        return mean_loss

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in self.val_loader:
                data = batch.to(self.device)
                recon, mean, logvar, z = self.model(data)
                total_loss += self.loss_fn(recon, data).item()
                n_batches += 1
        mean_loss = total_loss / n_batches
        print(f"====> Epoch: {epoch} Val loss: {mean_loss:.4f}")
        return mean_loss

    def run(self, n_epochs):
        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            self.epoch_train_losses.append(train_loss)
            self.epoch_val_losses.append(val_loss)

        # Plot losses at the end (minimal, automatic)
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Batch iteration losses
            if len(self.batch_losses) > 0:
                steps = [s for s, _ in self.batch_losses]
                losses = [l for _, l in self.batch_losses]
                axes[0].plot(steps, losses, linewidth=1)
            axes[0].set_title("Batch loss (iteration)")
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("MSE")

            # Epoch losses
            epochs = list(range(1, len(self.epoch_train_losses) + 1))
            axes[1].plot(epochs, self.epoch_train_losses, label="train", linewidth=2)
            axes[1].plot(epochs, self.epoch_val_losses, label="val", linewidth=2)
            axes[1].set_title("Epoch loss")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("MSE")
            axes[1].legend()

            plt.tight_layout()
            try:
                import io
                from IPython.display import Image as IPyImage, display as ipy_display
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                buf.seek(0)
                ipy_display(IPyImage(data=buf.getvalue()))
            except Exception:
                plt.show()
            plt.close(fig)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def scatter_latent_salsa(latents, labels=None, use_tsne=True, title="Salsa latent space"):
    """t-SNE or PCA scatter of latent codes. Labels optional."""
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
    except ImportError:
        raise ImportError("scatter_latent_salsa requires sklearn")
    import matplotlib.pyplot as plt

    latents = np.asarray(latents)
    if latents.ndim > 2:
        latents = latents.reshape(latents.shape[0], -1)

    if latents.shape[1] > 2 and use_tsne:
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, latents.shape[0] - 1))
        coords = reducer.fit_transform(latents)
    elif latents.shape[1] > 2:
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(latents)
    else:
        coords = latents[:, :2]

    fig, ax = plt.subplots(figsize=(8, 6))
    if labels is not None:
        ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", alpha=0.6, s=10)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=10)
    ax.set_title(title)
    plt.tight_layout()
    # Robust inline display (works even when matplotlib backend is non-interactive like Agg)
    try:
        import io
        from IPython.display import Image as IPyImage, display as ipy_display

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        ipy_display(IPyImage(data=buf.getvalue()))
    except Exception:
        plt.show()
    plt.close(fig)


def plot_motion_frame(seq, frame_idx=0, title=None):
    """Simple 1D plot of one frame's features (262 InterHuman or 263 HumanML3D)."""
    import matplotlib.pyplot as plt
    seq = np.asarray(seq)
    if seq.ndim == 3:
        seq = seq[0]
    frame = seq[frame_idx] if seq.ndim == 2 else seq
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(frame)
    ax.set_title(title or f"Frame {frame_idx}")
    ax.set_xlabel("Feature dim")
    plt.tight_layout()
    try:
        from IPython.display import display
        display(fig)
    except ImportError:
        plt.show()
    plt.close(fig)


def animate_skeleton_3d_gif(
    motion: np.ndarray,
    save_path: str = "salsa_motion.gif",
    representation_type: str = "interhuman",
    mean=None,
    std=None,
    title: str = "Salsa motion",
    fps: int = 20,
):
    """
    Animate 3D skeleton from motion data and save as GIF.
    Uses plot_3d_motion from Salsa motion representation (no changes to their code).

    Args:
        motion: (T, 262) for interhuman or (T, 263) for humanml3d. Can be normalized.
        save_path: Output path (.gif or .mp4)
        representation_type: 'interhuman' or 'humanml3d'
        mean, std: For denormalization (from dataset.mean, dataset.std). If None, assumes already denormalized.
        title: Plot title
        fps: Frames per second
    """
    motion = np.asarray(motion)
    if motion.ndim == 3:
        motion = motion[0]
    if mean is not None and std is not None:
        motion = motion * np.asarray(std) + np.asarray(mean)

    if representation_type == "interhuman":
        # First 66 dims = 22 joints * 3 (positions per frame)
        keypoints = motion[:, :66].reshape(len(motion), 22, 3).astype(np.float64)
        try:
            from in2in.utils.plot import plot_3d_motion as plot_3d
            from in2in.utils.paramUtil import HML_KINEMATIC_CHAIN
            kinematic_chain = HML_KINEMATIC_CHAIN
        except ImportError:
            from utils.motion_utils import plot_3d_motion as plot_3d
            from utils.paramUtil import t2m_kinematic_chain as kinematic_chain
        # in2IN plot expects mp_joints as list
        if "in2in" in str(plot_3d.__module__):
            plot_3d(save_path, kinematic_chain, [keypoints], title=title, fps=fps, radius=4)
        else:
            plot_3d(save_path, kinematic_chain, keypoints, title=title, fps=fps, radius=4)
    else:
        # HumanML3D: use recover_from_ric
        from utils.motion_utils import recover_from_ric, plot_3d_motion
        from utils.paramUtil import t2m_kinematic_chain
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        keypoints = recover_from_ric(
            torch.from_numpy(motion).float().to(device), 22
        ).cpu().numpy()
        plot_3d_motion(save_path, t2m_kinematic_chain, keypoints, title=title, fps=fps, radius=4)

    print(f"Saved: {save_path}")
    # Display GIF in notebook
    try:
        from IPython.display import Image, display
        display(Image(filename=save_path))
    except Exception:
        pass
