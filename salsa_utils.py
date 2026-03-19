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
    use_vqvae: bool = False,
    nb_code: int = 512,
    quantizer: str = "ema_reset",
    vq_mu: float = 0.99,
    commit_weight: float = 0.02,
):
    """
    Salsa GRU autoencoder matching motion_representation architecture.
    By default this is vanilla AE (use_vqvae=False). Set use_vqvae=True to enable
    VQ-VAE mode with defaults: nb_code=512, quantizer='ema_reset', vq_mu=0.99,
    commit_weight=0.02.
    """
    # MotionModel signatures vary across Salsa versions.
    # Build kwargs defensively and pass only supported args.
    import inspect

    base_kwargs = dict(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        latent_dim=latent_dim,
        seq_len=seq_len,
        dropout=dropout,
        encoder_type="gru",
        decoder_type="gru",
        use_vae=False,
        use_vqvae=use_vqvae,
        nb_code=nb_code,
        quantizer=quantizer,
        vq_mu=vq_mu,
    )
    sig = inspect.signature(MotionModel.__init__)
    supported = {k: v for k, v in base_kwargs.items() if k in sig.parameters}
    model = MotionModel(**supported)

    # keep workshop API default even if constructor doesn't expose it
    if hasattr(model, "commit_weight"):
        setattr(model, "commit_weight", commit_weight)
    return model


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
    """AE/VQ-VAE trainer for Salsa model (MSE + optional commit/velocity losses)."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        learning_rate=2e-4,
        device=None,
        use_vqvae: bool = False,
        commit_weight: float = 0.02,
        loss_vel_weight: float = 0.0,
        warm_up_epochs: int = 0,
        lr_scheduler_gamma: float = 0.05,
        use_multistep_scheduler: bool = False,
        lr_scheduler_milestones=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = float(learning_rate)
        self.use_vqvae = bool(use_vqvae)
        self.commit_weight = float(commit_weight)
        self.loss_vel_weight = float(loss_vel_weight)
        self.warm_up_epochs = int(warm_up_epochs)
        self.lr_scheduler_gamma = float(lr_scheduler_gamma)
        self.use_multistep_scheduler = bool(use_multistep_scheduler)

        # Match original trainer optimizer choice for VQ-VAE
        if self.use_vqvae:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=0.0,
                betas=(0.9, 0.99),
                eps=1e-8,
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=0.0,
            )

        self.scheduler = None
        if self.use_multistep_scheduler:
            milestones = lr_scheduler_milestones
            if milestones is None:
                milestones = [500, 1500]
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=milestones, gamma=self.lr_scheduler_gamma
            )

        # Loss tracking for visualization
        self.global_step = 0
        self.batch_losses = []  # list of (step, loss)
        self.epoch_train_losses = []  # list of floats
        self.epoch_val_losses = []  # list of floats

    def loss_fn(self, recon, x):
        return torch.nn.functional.mse_loss(recon, x)

    @staticmethod
    def _extract_outputs(model_out, use_vqvae=False):
        """Return reconstruction plus optional VQ terms from model output."""
        if not isinstance(model_out, (tuple, list)):
            return model_out, None, None
        recon = model_out[0]
        commit_loss = None
        perplexity = None
        if use_vqvae:
            # Expected VQ output in Salsa trainer: (recon, z, commit_loss, perplexity, code_idx)
            if len(model_out) > 2 and torch.is_tensor(model_out[2]):
                commit_loss = model_out[2]
            if len(model_out) > 3:
                perplexity = model_out[3]
        return recon, commit_loss, perplexity

    def _velocity_loss(self, recon, target):
        # Temporal velocity consistency: mse(diff_t(recon), diff_t(target))
        if recon.dim() < 3 or recon.size(1) < 2:
            return torch.tensor(0.0, device=recon.device)
        recon_vel = recon[:, 1:, :] - recon[:, :-1, :]
        target_vel = target[:, 1:, :] - target[:, :-1, :]
        return torch.nn.functional.mse_loss(recon_vel, target_vel)

    def _total_loss(self, recon, data, commit_loss=None):
        recon_loss = self.loss_fn(recon, data)
        vel_loss = self._velocity_loss(recon, data) if self.loss_vel_weight > 0.0 else torch.tensor(0.0, device=data.device)
        total = recon_loss
        if commit_loss is not None:
            total = total + self.commit_weight * commit_loss
        if self.loss_vel_weight > 0.0:
            total = total + self.loss_vel_weight * vel_loss
        return total, recon_loss, vel_loss

    def _warmup(self):
        if self.warm_up_epochs <= 0:
            return
        warm_up_iter = self.warm_up_epochs * len(self.train_loader)
        if warm_up_iter <= 1:
            return
        self.model.train()
        train_iter = iter(self.train_loader)
        for nb_iter in range(1, warm_up_iter):
            current_lr = self.learning_rate * (nb_iter + 1) / (warm_up_iter + 1)
            for g in self.optimizer.param_groups:
                g["lr"] = current_lr
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            data = batch.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(data)
            recon, commit_loss, _ = self._extract_outputs(out, use_vqvae=self.use_vqvae)
            loss, _, _ = self._total_loss(recon, data, commit_loss=commit_loss)
            loss.backward()
            self.optimizer.step()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in tqdm(self.train_loader, leave=False):
            data = batch.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(data)
            recon, commit_loss, _ = self._extract_outputs(out, use_vqvae=self.use_vqvae)
            loss, _, _ = self._total_loss(recon, data, commit_loss=commit_loss)
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
                out = self.model(data)
                recon, commit_loss, _ = self._extract_outputs(out, use_vqvae=self.use_vqvae)
                loss, _, _ = self._total_loss(recon, data, commit_loss=commit_loss)
                total_loss += loss.item()
                n_batches += 1
        mean_loss = total_loss / n_batches
        print(f"====> Epoch: {epoch} Val loss: {mean_loss:.4f}")
        return mean_loss

    def run(self, n_epochs):
        # Warm-up at beginning to mimic original trainer behavior
        self._warmup()
        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            self.epoch_train_losses.append(train_loss)
            self.epoch_val_losses.append(val_loss)
            if self.scheduler is not None:
                self.scheduler.step()

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
