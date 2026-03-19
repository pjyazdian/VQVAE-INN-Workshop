"""
Workshop helper module: data loaders, models, training, and visualization.
Used by all notebooks to keep them clean and reproducible.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import re
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

try:
    from PIL import Image
except ImportError:
    Image = None
try:
    import cv2
except ImportError:
    cv2 = None


# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------

def get_device():
    """Return torch device (cuda if available, else cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Data loaders
# -----------------------------------------------------------------------------

def get_mnist_loaders(root="./data", batch_size=32, num_workers=0):
    """
    MNIST train and validation DataLoaders.
    Images are normalized and returned as (B, 1, 28, 28). For linear AE, flatten in the trainer or model.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = torchvision.datasets.MNIST(
        root=root, train=True, download=True, transform=transform
    )
    val_set = torchvision.datasets.MNIST(
        root=root, train=False, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader

'''
class HandImageDataset(Dataset):
    """
    Load hand images from a directory. Each image is resized to img_size (default 128x128)
    and converted to grayscale (1 channel). Place images in root_dir (any .png, .jpg, .jpeg).
    """

    def __init__(self, image_paths, img_size=(128, 128), transform=None):
        raise NotImplementedError(
            "HandImageDataset (hand-image-only training helpers) was disabled for this workshop cleanup. "
            "Use get_hand_csv_loaders + plot_hand_keypoints_on_image for visualization, and train on keypoints only."
        )
        self.img_size = img_size
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        if Image is None:
            raise ImportError("PIL is required for HandImageDataset. pip install Pillow")
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0  # no label by default
'''

def get_hand_image_loaders(root_dir, batch_size=32, val_ratio=0.1, img_size=(128, 128), num_workers=0):
    """
    Build train/val loaders for hand images under root_dir.
    Expects image files (.png, .jpg, .jpeg). Uses first (1 - val_ratio) for train.
    """
    raise NotImplementedError(
        "get_hand_image_loaders (hand-image-only training helpers) was disabled for this workshop cleanup. "
        "Use get_hand_csv_loaders for visualization and train on keypoints only."
    )
    if not root.exists():
        raise FileNotFoundError(f"Hand image root not found: {root_dir}")
    exts = {".png", ".jpg", ".jpeg"}
    paths = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"No images found under {root_dir}")
    n_val = max(1, int(len(paths) * val_ratio))
    n_train = len(paths) - n_val
    train_paths, val_paths = paths[:n_train], paths[n_train:]
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = HandImageDataset(train_paths, img_size=img_size, transform=transform)
    val_set = HandImageDataset(val_paths, img_size=img_size, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


class HandKeypointDataset(Dataset):
    """
    63-dim hand keypoints (21 landmarks x 3 coords). Load from CSV or .npy.
    CSV: 63 columns (or 64 with label in last column). No header or header row.
    .npy: array of shape (N, 63) or (N, 64) with optional label in last column.
    """

    def __init__(self, data, labels=None):
        self.data = torch.as_tensor(np.asarray(data), dtype=torch.float32)
        if self.data.dim() == 1:
            self.data = self.data.unsqueeze(0)
        if self.data.size(-1) == 64 and labels is None:
            labels = self.data[:, -1].long()
            self.data = self.data[:, :63]
        self.labels = labels if labels is not None else torch.zeros(self.data.size(0), dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_hand_csv_loaders(csv_path, image_root, letters=None, batch_size=32, val_ratio=0.2, img_size=(128, 128), num_workers=0):
    """
    Load hand data from CSV + raw images. CSV must have:
    - Column 0: path (relative to image_root), e.g. "A-samples/0.jpg"
    - Column 1: label (letter, e.g. A, B, ...)
    - Columns 2–64: 63 floats (21 landmarks × 3, flattened)
    Landmarks are assumed to come from MediaPipe (hand landmarker); extraction code is not shared.
    If letters is not None (e.g. ['A','B','C']), only those classes are used for training/inference.
    Returns: (train_loader_img, val_loader_img, train_loader_kp, val_loader_kp).
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    path_col = df.columns[0]
    label_col = df.columns[1]
    lm_cols = list(df.columns[2:2+63])
    if len(lm_cols) != 63:
        raise ValueError(f"CSV must have 63 landmark columns after path and label; got {len(lm_cols)}.")
    if letters is not None:
        letters = [str(l).upper() for l in letters]
        df = df[df[label_col].astype(str).str.upper().isin(letters)].copy()
    if len(df) == 0:
        raise ValueError("No rows left after filtering by letters. Check CSV and LETTERS.")
    paths = df[path_col].astype(str).tolist()
    keypoints = df[lm_cols].values.astype(np.float32)
    labels = df[label_col].astype(str).str.upper()
    uniq = sorted(labels.unique())
    label_to_idx = {u: i for i, u in enumerate(uniq)}
    labels_idx = np.array([label_to_idx[l] for l in labels])
    n = len(paths)
    idx = np.random.RandomState(42).permutation(n)
    n_val = max(1, int(n * val_ratio))
    train_idx, val_idx = idx[: n - n_val], idx[n - n_val:]
    root = Path(image_root)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    class _HandCsvDataset(Dataset):
        def __init__(self, indices, mode):
            self.indices = indices
            self.mode = mode  # 'image' or 'keypoint'
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            j = self.indices[i]
            lbl = labels_idx[j]
            if self.mode == "keypoint":
                return torch.as_tensor(keypoints[j], dtype=torch.float32), lbl
            p = root / paths[j]
            if not p.exists():
                raise FileNotFoundError(f"Image not found: {p}")
            if Image is None:
                raise ImportError("PIL is required. pip install Pillow")
            img = Image.open(p).convert("RGB")
            img = transform(img)
            return img, lbl

    train_set_img = _HandCsvDataset(train_idx, "image")
    val_set_img = _HandCsvDataset(val_idx, "image")
    train_set_kp = _HandCsvDataset(train_idx, "keypoint")
    val_set_kp = _HandCsvDataset(val_idx, "keypoint")
    train_loader_img = DataLoader(train_set_img, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader_img = DataLoader(val_set_img, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_loader_kp = DataLoader(train_set_kp, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader_kp = DataLoader(val_set_kp, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader_img, val_loader_img, train_loader_kp, val_loader_kp


def get_hand_keypoint_loaders(csv_path=None, npy_path=None, batch_size=32, val_ratio=0.1, num_workers=0):
    """
    Build train/val loaders for 63-dim keypoints. Provide either csv_path or npy_path.
    """
    if csv_path is not None:
        import pandas as pd
        df = pd.read_csv(csv_path, header=None)
        data = df.values.astype(np.float32)
    elif npy_path is not None:
        data = np.load(npy_path).astype(np.float32)
    else:
        raise ValueError("Provide csv_path or npy_path")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n = len(data)
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val
    train_data, val_data = data[:n_train], data[n_train:]
    train_set = HandKeypointDataset(train_data)
    val_set = HandKeypointDataset(val_data)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class LinearAE(nn.Module):
    """
    Linear autoencoder for flattened vectors (e.g. MNIST 784 or hand keypoints 63).
    Architecture: input_dim -> hidden_dim -> latent_dim -> hidden_dim -> input_dim.

    Optional VQ-VAE (uses ``vq_layer.VQVAE``): set ``use_vq=True``.
    Default VQ settings match workshop config: nb_code=512, quantizer='ema_reset',
    vq_mu=0.99, commit_weight=0.02.

    When ``use_vq`` is True, ``forward`` returns a 4-tuple:
    ``(recon, commit_loss, perplexity, code_idx)`` for training.
    Use ``reconstruct(x)`` for a single reconstruction tensor in eval/notebooks.
    """

    def __init__(
        self,
        latent_dim=2,
        input_dim=784,
        hidden_dim=1024,
        use_vq=False,
        nb_code=512,
        quantizer="ema_reset",
        vq_mu=0.99,
        commit_weight=0.02,
        vq_beta=1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.use_vq = use_vq
        self.commit_weight = float(commit_weight)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )
        if use_vq:
            from vq_layer import VQVAE

            self.vq = VQVAE(
                nb_code=nb_code,
                code_dim=latent_dim,
                quantizer=quantizer,
                mu=vq_mu,
                beta=vq_beta,
            )
        else:
            self.vq = None

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def quantize_indices(self, x):
        """
        Map inputs to VQ code indices (batch,). Returns ``None`` if ``use_vq`` is False.
        """
        if not self.use_vq:
            return None
        z = self.encode(x)
        return self.vq.quantize(z)

    def reconstruct(self, x):
        """Reconstruction tensor only (works with or without VQ)."""
        if not self.use_vq:
            return self.forward(x)
        z = self.encode(x)
        z_q, _, _, _ = self.vq(z)
        return self.decode(z_q)

    def forward(self, x):
        z = self.encode(x)
        if not self.use_vq:
            return self.decode(z)
        z_q, commit_loss, perplexity, code_idx = self.vq(z)
        recon = self.decode(z_q)
        return recon, commit_loss, perplexity, code_idx

'''
class CNN_AE(nn.Module):
    """
    CNN autoencoder for images (B, 1, H, W). Built for 28x28 MNIST; can adapt for 128x128.
    Encoder: Conv2d -> flatten -> linear -> latent. Decoder: linear -> view -> ConvTranspose2d.
    """

    def __init__(self, latent_dim=32, in_channels=1, img_size=28):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        # Encoder: 28 -> 14 -> 7
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.enc_flat_size = 64 * (img_size // 4) * (img_size // 4)
        self.enc_linear = nn.Linear(self.enc_flat_size, latent_dim)
        self.dec_linear = nn.Linear(latent_dim, self.enc_flat_size)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.view(h.size(0), -1)
        return self.enc_linear(h)

    def decode(self, z):
        h = self.dec_linear(z)
        h = h.view(z.size(0), 64, self.img_size // 4, self.img_size // 4)
        return self.dec(h)

    def forward(self, x):
        return self.decode(self.encode(x))
'''

class GRU_AE(nn.Module):
    """
    GRU autoencoder for sequences (B, T, F) e.g. (B, 30, 63).
    Based on motion-sequence AE: pre-linear, GRU, intermediate FC, latent.
    No dropout, unidirectional.

    Encoder: fc_in(input_dim->hidden_dim) -> GRU -> fc_intermediate -> fc_out -> latent_dim
    Decoder: latent -> initial hidden; autoregressive GRU with first frame (or zeros).
    """

    def __init__(self, seq_len=30, input_dim=63, hidden_dim=64, latent_dim=32, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder: pre-linear (input -> hidden), GRU in hidden space, intermediate -> latent
        self.encoder_fc_in = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.encoder_gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.encoder_fc_intermediate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.encoder_fc_out = nn.Linear(hidden_dim, latent_dim)

        # Decoder: latent -> initial hidden state; autoregressive GRU
        self.decoder_input_to_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim * num_layers),
        )
        self.decoder_fc_frame = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.decoder_gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.decoder_fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        x = self.encoder_fc_in(x)  # (B, T, hidden_dim)
        _, h = self.encoder_gru(x)
        h = h[-1]  # last layer (B, hidden_dim)
        h = self.encoder_fc_intermediate(h)
        return self.encoder_fc_out(h)

    def decode(self, z, first_frame=None):
        batch_size = z.size(0)
        hidden = self.decoder_input_to_hidden(z)
        hidden = hidden.view(batch_size, self.num_layers, self.hidden_dim)
        hidden = hidden.permute(1, 0, 2).contiguous()

        if first_frame is None:
            decoder_input = torch.zeros(batch_size, 1, self.input_dim, device=z.device)
        else:
            decoder_input = first_frame.unsqueeze(1)
        decoder_input = self.decoder_fc_frame(decoder_input)

        outputs = []
        for _ in range(self.seq_len):
            gru_out, hidden = self.decoder_gru(decoder_input, hidden)
            out = self.decoder_fc_out(gru_out)
            outputs.append(out)
            decoder_input = self.decoder_fc_frame(out)

        return torch.cat(outputs, dim=1)

    def forward(self, x, first_frame=None):
        z = self.encode(x)
        if first_frame is None:
            first_frame = x[:, 0, :]
        return self.decode(z, first_frame)


class SequenceDataset(Dataset):
    """Dataset for (N, T, F) sequences; returns (seq, 0)."""

    def __init__(self, data):
        self.data = torch.as_tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], 0


# Canonical gesture classes (fixed order); raw labels are normalized and matched to these.
CANONICAL_GESTURE_CLASSES = [
    "Left Swipe",
    "Right Swipe",
    "Stop Gesture",
    "Thumbs Down",
    "Thumbs Up",
]


def _normalize_for_match(s: str) -> str:
    """Lowercase, replace '-' with ' ', collapse spaces."""
    s = s.lower().replace("-", " ").strip()
    return " ".join(s.split())


def _extract_gesture_label(key: str) -> str:
    """Extract gesture name from key like 'train/WIN_..._Pro_Right Swipe_new' -> 'Right Swipe'."""
    name = key.split("/")[-1] if "/" in key else key
    m = re.search(r"_Pro_(.+?)_new", name)
    raw = m.group(1).strip() if m else name
    raw_norm = _normalize_for_match(raw)
    for canonical in CANONICAL_GESTURE_CLASSES:
        if _normalize_for_match(canonical) == raw_norm:
            return canonical
    return CANONICAL_GESTURE_CLASSES[0]  # fallback if unknown


def _frame_number(path: Path) -> int:
    """Extract frame index from filename, e.g. WIN_..._Pro_00036.png -> 36."""
    m = re.search(r"_(\d+)\.(?:png|jpg|jpeg)$", path.name, re.I)
    return int(m.group(1)) if m else 0


def _sorted_frame_paths(folder: Path, num_frames: int = 30) -> list[Path]:
    """Return up to num_frames image paths in folder, sorted by frame number."""
    paths = [p for p in folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    paths = sorted(paths, key=_frame_number)[:num_frames]
    return paths


class HandGestureSequenceDataset(Dataset):
    """
    Dataset for Hand Gesture data: each sample = (30 frames RGB, 30×21×3 keypoints, class label).
    Frames and keypoints come from the same sequence; train/val split from folder structure.
    """

    def __init__(
        self,
        data_root: str | Path,
        landmarks_path: str | Path,
        split: str,
        img_size=(128, 128),
        num_frames=30,
    ):
        """
        data_root: path to data/Hand_Gesture/data (contains train/ and val/)
        landmarks_path: path to hand_gesture_landmarks.npy
        split: 'train' or 'val'
        img_size: (H, W) to resize frames
        num_frames: 30
        """
        self.data_root = Path(data_root)
        self.landmarks_path = Path(landmarks_path)
        self.split = split
        self.img_size = img_size
        self.num_frames = num_frames

        landmarks = np.load(self.landmarks_path, allow_pickle=True).item()
        split_prefix = f"{split}/"
        self.samples = []  # (folder_path, key, canonical_label)
        self.label2idx = {lbl: i for i, lbl in enumerate(CANONICAL_GESTURE_CLASSES)}
        for key in landmarks:
            if not key.startswith(split_prefix):
                continue
            folder_name = key[len(split_prefix) :]
            folder_path = self.data_root / split / folder_name
            if not folder_path.is_dir():
                continue
            canonical_label = _extract_gesture_label(key)
            self.samples.append((folder_path, key, canonical_label))
        self.landmarks = landmarks

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_path, key, label_str = self.samples[idx]
        label_idx = self.label2idx[label_str]

        # Load frames (30, H, W, 3) -> (30, 3, H, W)
        frame_paths = _sorted_frame_paths(folder_path, self.num_frames)
        frames = []
        for p in frame_paths:
            if Image is not None:
                img = Image.open(p).convert("RGB")
                img = np.array(img)
            elif cv2 is not None:
                img = cv2.imread(str(p))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                raise ImportError("PIL or cv2 required for HandGestureSequenceDataset")
            if self.img_size:
                if Image is not None:
                    img = np.array(Image.fromarray(img).resize((self.img_size[1], self.img_size[0])))
                else:
                    img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
            frames.append(img)
        # Pad if fewer than num_frames
        while len(frames) < self.num_frames:
            frames.append(frames[-1].copy() if frames else np.zeros((*self.img_size, 3), dtype=np.uint8))
        frames = np.stack(frames[: self.num_frames], axis=0)  # (T, H, W, 3)
        frames = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0  # (T, 3, H, W)

        # Keypoints (30, 21, 3) -> (30, 63) for GRU
        kp = self.landmarks[key].astype(np.float32)  # (30, 21, 3)
        kp = torch.from_numpy(kp.reshape(self.num_frames, -1))  # (30, 63)

        # Return (keypoints, frames, label) so batch[0] is keypoints for AETrainer
        return kp, frames, label_idx


def get_hand_gesture_sequence_loaders(
    data_root: str | Path,
    landmarks_path: str | Path,
    img_size=(128, 128),
    num_frames=30,
    batch_size=32,
    num_workers=0,
):
    """
    Train and val loaders for Hand Gesture data.
    Each batch: (keypoints, frames, labels) — keypoints first for AETrainer compatibility.
    - keypoints: (B, 30, 63)
    - frames: (B, 30, 3, H, W)
    - labels: (B,)
    """
    data_root = Path(data_root)
    landmarks_path = Path(landmarks_path)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not landmarks_path.exists():
        raise FileNotFoundError(f"Landmarks not found: {landmarks_path}")

    train_set = HandGestureSequenceDataset(
        data_root=data_root,
        landmarks_path=landmarks_path,
        split="train",
        img_size=img_size,
        num_frames=num_frames,
    )
    val_set = HandGestureSequenceDataset(
        data_root=data_root,
        landmarks_path=landmarks_path,
        split="val",
        img_size=img_size,
        num_frames=num_frames,
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, train_set.label2idx


class EgoGestureSequenceDataset(Dataset):
    """
    Dataset for Ego Gesture v4: skeleton-only, each sample = (keypoints, dummy_frames, label).
    Uses first hand only (63 dims). Frames are zeros (no RGB in this dataset).
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        num_frames=30,
        img_size=(128, 128),
    ):
        """
        data_root: path to ego_gesture_v4 (contains skeletons/, split_files/)
        split: 'train' or 'val'
        num_frames: pad/truncate to this length
        img_size: for dummy frames shape
        """
        self.data_root = Path(data_root)
        self.split = split
        self.num_frames = num_frames
        self.img_size = img_size

        split_file = self.data_root / "split_files" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        self.samples = []
        with open(split_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue
                rel_path, label_str = parts[0], parts[1]
                label = int(label_str)
                self.samples.append((rel_path, label))
        self.label2idx = {i: i for i in range(83)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        skel_path = self.data_root / rel_path
        if not skel_path.exists():
            raise FileNotFoundError(f"Skeleton not found: {skel_path}")

        # Load skeleton: 126 floats per line (left 63 + right 63), use first hand only
        rows = []
        with open(skel_path) as f:
            for line in f:
                vals = [float(x) for x in line.strip().split(",")]
                if len(vals) >= 63:
                    rows.append(vals[:63])
        kp = np.array(rows, dtype=np.float32) if rows else np.zeros((1, 63), dtype=np.float32)

        # Pad or truncate to num_frames
        T = kp.shape[0]
        if T >= self.num_frames:
            kp = kp[: self.num_frames]
        else:
            kp = np.concatenate([kp, np.tile(kp[-1:], (self.num_frames - T, 1))], axis=0)

        kp = torch.from_numpy(kp)
        # Dummy frames (no RGB in Ego_Gesture)
        frames = torch.zeros(self.num_frames, 3, self.img_size[0], self.img_size[1], dtype=torch.float32)
        return kp, frames, label


def get_ego_gesture_sequence_loaders(
    data_root: str | Path,
    num_frames=30,
    img_size=(128, 128),
    batch_size=32,
    num_workers=0,
):
    """
    Train and val loaders for Ego Gesture v4 (skeleton only, first hand).
    Each batch: (keypoints, frames, labels) — frames are zeros.
    """
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    train_set = EgoGestureSequenceDataset(
        data_root=data_root,
        split="train",
        num_frames=num_frames,
        img_size=img_size,
    )
    val_set = EgoGestureSequenceDataset(
        data_root=data_root,
        split="val",
        num_frames=num_frames,
        img_size=img_size,
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    label2idx = {i: i for i in range(83)}
    return train_loader, val_loader, label2idx


def get_sequence_loaders(npy_path=None, seq_len=20, keypoint_dim=63, batch_size=32, val_ratio=0.1, num_workers=0):
    """
    Load sequences (N, T, F) from .npy. If 2D (N*T, F), reshape to (N, seq_len, F).
    """
    if npy_path is None or not Path(npy_path).exists():
        return None, None
    data = np.load(npy_path).astype(np.float32)
    if data.ndim == 2:
        n_total = data.shape[0]
        n_seq = n_total // seq_len
        data = data[: n_seq * seq_len].reshape(n_seq, seq_len, -1)
    n = len(data)
    n_val = max(1, int(n * val_ratio))
    train_set = SequenceDataset(data[: n - n_val])
    val_set = SequenceDataset(data[n - n_val:])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

class AETrainer:
    """
    Trainer for autoencoder: MSE reconstruction loss, Adam.
    flatten_input=True: data is flattened to (B, D) for LinearAE.
    flatten_input=False: data kept as (B, 1, H, W) or (B, T, F) for CNN_AE / GRU_AE.

    VQ-VAE (``LinearAE(use_vq=True)``): ``forward`` returns
    ``(recon, commit_loss, perplexity, code_idx)``. Total loss is
    ``MSE(recon, x) + commit_weight * commit_loss``.
    If ``commit_weight`` is None, uses ``model.commit_weight`` when present, else 0.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        learning_rate=1e-3,
        device=None,
        flatten_input=True,
        commit_weight=None,
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.flatten_input = flatten_input
        if commit_weight is None:
            commit_weight = float(getattr(model, "commit_weight", 0.0))
        self.commit_weight = float(commit_weight)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )

    def loss_fn(self, recon, x):
        return F.mse_loss(recon, x)

    @staticmethod
    def _unpack_forward(out):
        """Return (recon, commit_loss_or_None)."""
        if isinstance(out, tuple) and len(out) == 4:
            recon, commit_loss, _, _ = out
            return recon, commit_loss
        return out, None

    def _prepare_batch(self, batch):
        data = batch[0] if isinstance(batch, (list, tuple)) else batch
        data = data.to(self.device)
        if self.flatten_input:
            data = data.view(data.size(0), -1)
        return data

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in tqdm(self.train_loader, leave=False):
            data = self._prepare_batch(batch)
            self.optimizer.zero_grad()
            recon, commit_loss = self._unpack_forward(self.model(data))
            loss = self.loss_fn(recon, data)
            if commit_loss is not None:
                loss = loss + self.commit_weight * commit_loss
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        mean_loss = total_loss / n_batches
        print(f"====> Epoch: {epoch} Train loss: {mean_loss:.4f}")
        return mean_loss

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in self.val_loader:
                data = self._prepare_batch(batch)
                recon, commit_loss = self._unpack_forward(self.model(data))
                loss = self.loss_fn(recon, data)
                if commit_loss is not None:
                    loss = loss + self.commit_weight * commit_loss
                total_loss += loss.item()
                n_batches += 1
        mean_loss = total_loss / n_batches
        print(f"====> Epoch: {epoch} Val loss: {mean_loss:.4f}")
        return mean_loss

    def run(self, n_epochs):
        for epoch in range(1, n_epochs + 1):
            self.train_epoch(epoch)
            self.validate(epoch)


# -----------------------------------------------------------------------------
# VQ component and VQ-VAE
# -----------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    """
    Vector quantization layer: map continuous z to nearest codebook entry.
    codebook: (K, D). Returns quantized z, indices (B,), and commitment loss term.
    """

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z):
        # z: (B, D)
        z_flat = z.view(-1, self.embedding_dim)
        distances = torch.cdist(z_flat, self.embedding.weight)
        indices = distances.argmin(dim=1)
        indices = indices.view(*z.shape[:-1])
        z_q = self.embedding(indices)
        # Straight-through: gradient of z_q flows as z
        z_q_sg = z + (z_q - z).detach()
        return z_q_sg, indices, z_q


class VQVAE_Linear(nn.Module):
    """
    VQ-VAE with linear encoder/decoder for flattened 784 input.
    encoder -> z_e -> VQ -> z_q -> decoder -> recon.
    """

    def __init__(self, latent_dim=32, num_embeddings=512, input_dim=784, hidden_dim=1024):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.vq = VectorQuantizer(num_embeddings, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z_e = self.encode(x)
        z_q, indices, z_q_embed = self.vq(z_e)
        recon = self.decode(z_q)
        return recon, z_e, z_q_embed, indices


class VQVAETrainer:
    """
    Trainer for VQ-VAE: reconstruction loss + commitment loss (beta * ||z_e - z_q||^2).
    """

    def __init__(self, model, train_loader, val_loader, learning_rate=1e-3, device=None, beta=1.0):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.beta = beta
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

    def loss_fn(self, recon, x, z_e, z_q):
        recon_loss = F.mse_loss(recon, x)
        commitment_loss = F.mse_loss(z_e, z_q)
        return recon_loss + self.beta * commitment_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in tqdm(self.train_loader, leave=False):
            data = batch[0] if isinstance(batch, (list, tuple)) else batch
            data = data.to(self.device).view(data.size(0), -1)
            self.optimizer.zero_grad()
            recon, z_e, z_q, _ = self.model(data)
            loss = self.loss_fn(recon, data, z_e, z_q)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"====> Epoch: {epoch} Train loss: {total_loss / n_batches:.4f}")
        return total_loss / n_batches

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in self.val_loader:
                data = batch[0] if isinstance(batch, (list, tuple)) else batch
                data = data.to(self.device).view(data.size(0), -1)
                recon, z_e, z_q, _ = self.model(data)
                total_loss += self.loss_fn(recon, data, z_e, z_q).item()
                n_batches += 1
        print(f"====> Epoch: {epoch} Val loss: {total_loss / n_batches:.4f}")
        return total_loss / n_batches

    def run(self, n_epochs):
        for epoch in range(1, n_epochs + 1):
            self.train_epoch(epoch)
            self.validate(epoch)


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

# MediaPipe-style hand skeleton: 21 landmarks, connections as (i, j) index pairs.
# Defined here so the workshop does not require the mediapipe package.
HAND_CONNECTIONS = (
    (0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17),  # palm
    (1, 2), (2, 3), (3, 4),                                # thumb
    (5, 6), (6, 7), (7, 8),                                # index
    (9, 10), (10, 11), (11, 12),                           # middle
    (13, 14), (14, 15), (15, 16),                          # ring
    (17, 18), (18, 19), (19, 20),                          # pinky
)


def plot_3d_hand_landmarks(landmarks, title=None):
    """
    Interactive 3D plot of hand landmarks (21 points) with skeleton connections.
    landmarks: array of shape (21, 3) or (63,) — x,y,z per landmark.
    Requires plotly: pip install plotly.
    """
    raise NotImplementedError(
        "Plotly interactive 3D hand plotting was disabled for this workshop cleanup. "
        "Use plot_hand_keypoints_on_image (2D overlay) instead."
    )
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plot_3d_hand_landmarks requires plotly. Install with: pip install plotly")
    landmarks = np.asarray(landmarks)
    if landmarks.size == 63:
        landmarks = landmarks.reshape(21, 3)
    if landmarks.shape != (21, 3):
        raise ValueError("landmarks must be (21, 3) or (63,), got %s" % (landmarks.shape,))
    x, y, z = landmarks[:, 0], landmarks[:, 1], landmarks[:, 2]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers+text",
        marker=dict(size=5, color="blue"),
        text=[str(i) for i in range(21)],
        textposition="top center",
    ))
    for (i, j) in HAND_CONNECTIONS:
        fig.add_trace(go.Scatter3d(
            x=[x[i], x[j]], y=[y[i], y[j]], z=[z[i], z[j]],
            mode="lines",
            line=dict(color="red", width=2),
        ))
    fig.update_layout(
        title=title or "3D Hand Landmarks",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="auto",
        ),
    )
    fig.show()


def plot_3d_hand_landmarks_pair(original, reconstructed, title="Original vs Reconstructed"):
    """
    Interactive 3D plot of two hands (e.g. original and reconstructed) in the same scene.
    original, reconstructed: (21, 3) or (63,) each.
    """
    raise NotImplementedError(
        "Plotly interactive 3D hand plotting was disabled for this workshop cleanup. "
        "Use plot_hand_keypoints_on_image (2D overlay) instead."
    )
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plot_3d_hand_landmarks_pair requires plotly. Install with: pip install plotly")
    def to_21_3(a):
        a = np.asarray(a)
        return a.reshape(21, 3) if a.size == 63 else a
    orig = to_21_3(original)
    recon = to_21_3(reconstructed)
    fig = go.Figure()
    for name, pts, color in [("Original", orig, "blue"), ("Reconstructed", recon, "orange")]:
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers+text",
            marker=dict(size=5, color=color),
            text=[str(i) for i in range(21)],
            textposition="top center",
            name=name,
        ))
        for (i, j) in HAND_CONNECTIONS:
            fig.add_trace(go.Scatter3d(
                x=[x[i], x[j]], y=[y[i], y[j]], z=[z[i], z[j]],
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False,
            ))
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="auto",
        ),
    )
    fig.show()


def animate_sequence_3d(seq, title=None):
    """
    Animate a sequence of hand landmarks over time (interactive slider).
    seq: (T, 63) or (T, 21, 3) — T frames of 21 landmarks.
    Camera fixed to image-like view (from above xy plane); axes fixed from full-sequence
    min/max so the hand moves visibly when scrubbing (not stuck in center).
    Requires plotly: pip install plotly.
    """
    raise NotImplementedError(
        "Plotly interactive 3D hand animation was disabled for this workshop cleanup. "
        "Use animate_sequence_dual (2D overlay) or a GIF-based visualization instead."
    )
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("animate_sequence_3d requires plotly. Install with: pip install plotly")
    seq = np.asarray(seq, dtype=np.float64)
    if seq.ndim == 2 and seq.shape[1] == 63:
        seq = seq.reshape(-1, 21, 3)
    if seq.ndim != 3 or seq.shape[1] != 21 or seq.shape[2] != 3:
        raise ValueError("seq must be (T, 63) or (T, 21, 3), got %s" % (seq.shape,))
    # Flip y so 3D view matches image (image y down -> 3D y up)
    seq = seq.copy()
    seq[:, :, 1] = 1.0 - seq[:, :, 1]
    T = seq.shape[0]
    # Fixed axes matching image coordinate space: x,y in [0,1], z for depth
    # This keeps the 3D box stable frame-to-frame (no axis drift)
    scene_range = dict(
        xaxis=dict(range=[0, 1], autorange=False),
        yaxis=dict(range=[0, 1], autorange=False),
        zaxis=dict(range=[-0.15, 0.15], autorange=False),
    )
    # Camera from above xy plane (image-like view); center at domain center
    scene_camera = dict(
        eye=dict(x=0, y=0, z=2),
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=1, z=0),
    )
    frames = []
    for t in range(T):
        pts = seq[t]
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        trace_pts = go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers+text",
            marker=dict(size=5, color="blue"),
            text=[str(i) for i in range(21)],
            textposition="top center",
        )
        trace_lines = []
        for (i, j) in HAND_CONNECTIONS:
            trace_lines.append(
                go.Scatter3d(
                    x=[x[i], x[j]], y=[y[i], y[j]], z=[z[i], z[j]],
                    mode="lines",
                    line=dict(color="red", width=2),
                )
            )
        frames.append(go.Frame(data=[trace_pts] + trace_lines, name=str(t)))
    fig = go.Figure(
        data=frames[0]["data"],
        frames=frames,
        layout=go.Layout(
            title=title or "Sequence (use slider to scrub)",
            scene=dict(
                xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=0.3),
                camera=scene_camera,
                **scene_range,
            ),
            updatemenus=[
                {
                    "buttons": [
                        {"args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}],
                         "label": "Play", "method": "animate"},
                        {"args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                         "label": "Pause", "method": "animate"},
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1, "y": 0,
                }
            ],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "y": 0,
                "xanchor": "left",
                "x": 0.1,
                "len": 0.9,
                "currentvalue": {"prefix": "Frame: ", "visible": True},
                "pad": {"b": 10, "t": 50},
                "steps": [{"args": [[f.name], {"frame": {"duration": 0}, "mode": "immediate"}],
                          "label": str(t), "method": "animate"} for t, f in enumerate(frames)],
            }],
        ),
    )
    fig.show()


def animate_sequence_dual(frames, keypoints, title=None, fps=15, save_path=None):
    """
    Create and display a playable video: left = raw frames, right = frames with landmarks overlaid.
    Saves as GIF (playable in Jupyter/browsers) or MP4 if save_path ends with .mp4.
    frames: (T, 3, H, W) tensor or (T, H, W, 3) array, float 0-1 or uint8.
    keypoints: (T, 63) or (T, 21, 3) — normalized x,y for overlay.
    save_path: where to save (default: temp .gif; use .mp4 for video).
    Requires cv2.
    """
    import tempfile
    import os
    if cv2 is None:
        raise ImportError("animate_sequence_dual requires opencv-python")
    frames = np.asarray(frames)
    keypoints = np.asarray(keypoints)
    if torch.is_tensor(frames):
        frames = frames.cpu().numpy()
    if frames.ndim == 4 and frames.shape[1] == 3:
        frames = np.transpose(frames, (0, 2, 3, 1))  # (T, H, W, 3)
    if frames.dtype != np.uint8 or frames.max() <= 1:
        frames = (np.clip(frames, 0, 1) * 255).astype(np.uint8)
    if keypoints.ndim == 2 and keypoints.shape[1] == 63:
        keypoints = keypoints.reshape(-1, 21, 3)
    T = min(len(frames), len(keypoints))
    overlay_frames = []
    for t in range(T):
        img = draw_hand_landmarks_on_image(
            frames[t].copy(), keypoints[t],
            color_connections=(255, 100, 100), color_points=(0, 200, 255),
        )
        overlay_frames.append(img)
    side_by_side = np.hstack([frames, overlay_frames])  # (T, H, W*2, 3)
    if save_path is None:
        fd, save_path = tempfile.mkstemp(suffix=".gif")
        os.close(fd)
    use_gif = str(save_path).lower().endswith(".gif")
    if use_gif:
        try:
            import imageio
            imageio.mimsave(save_path, side_by_side, fps=fps, loop=0)
        except ImportError:
            from PIL import Image
            imgs = [Image.fromarray(side_by_side[t]) for t in range(T)]
            imgs[0].save(save_path, save_all=True, append_images=imgs[1:], duration=1000 // fps, loop=0)
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(save_path, fourcc, fps, (side_by_side.shape[2], side_by_side.shape[1]))
        for t in range(T):
            frame_bgr = cv2.cvtColor(side_by_side[t], cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()
    try:
        from IPython.display import Image as IPImage, display
        if use_gif:
            with open(save_path, "rb") as f:
                display(IPImage(data=f.read()))
        else:
            from IPython.display import Video
            display(Video(save_path, embed=True))
    except ImportError:
        print(f"Saved to {save_path}. Open it to play.")


def animate_reconstruction_comparison(frames, orig_keypoints, recon_keypoints, title=None, fps=15, save_path=None):
    """
    Side-by-side video: left = frames + ground-truth landmarks, right = frames + reconstructed landmarks.
    frames: (T, 3, H, W) tensor or (T, H, W, 3) array.
    orig_keypoints, recon_keypoints: (T, 63) or (T, 21, 3).
    Saves as GIF by default (playable in Jupyter).
    """
    import tempfile
    import os
    if cv2 is None:
        raise ImportError("animate_reconstruction_comparison requires opencv-python")
    frames = np.asarray(frames)
    orig_keypoints = np.asarray(orig_keypoints)
    recon_keypoints = np.asarray(recon_keypoints)
    if torch.is_tensor(frames):
        frames = frames.cpu().numpy()
    if frames.ndim == 4 and frames.shape[1] == 3:
        frames = np.transpose(frames, (0, 2, 3, 1))
    if frames.dtype != np.uint8 or frames.max() <= 1:
        frames = (np.clip(frames, 0, 1) * 255).astype(np.uint8)
    if orig_keypoints.ndim == 2 and orig_keypoints.shape[1] == 63:
        orig_keypoints = orig_keypoints.reshape(-1, 21, 3)
    if recon_keypoints.ndim == 2 and recon_keypoints.shape[1] == 63:
        recon_keypoints = recon_keypoints.reshape(-1, 21, 3)
    T = min(len(frames), len(orig_keypoints), len(recon_keypoints))
    orig_overlay = []
    recon_overlay = []
    for t in range(T):
        orig_overlay.append(draw_hand_landmarks_on_image(
            frames[t].copy(), orig_keypoints[t],
            color_connections=(255, 100, 100), color_points=(0, 200, 255),
        ))
        recon_overlay.append(draw_hand_landmarks_on_image(
            frames[t].copy(), recon_keypoints[t],
            color_connections=(255, 180, 100), color_points=(255, 165, 0),
        ))
    side_by_side = np.hstack([np.array(orig_overlay), np.array(recon_overlay)])
    if save_path is None:
        fd, save_path = tempfile.mkstemp(suffix=".gif")
        os.close(fd)
    use_gif = str(save_path).lower().endswith(".gif")
    if use_gif:
        try:
            import imageio
            imageio.mimsave(save_path, side_by_side, fps=fps, loop=0)
        except ImportError:
            from PIL import Image
            imgs = [Image.fromarray(side_by_side[t]) for t in range(T)]
            imgs[0].save(save_path, save_all=True, append_images=imgs[1:], duration=1000 // fps, loop=0)
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(save_path, fourcc, fps, (side_by_side.shape[2], side_by_side.shape[1]))
        for t in range(T):
            frame_bgr = cv2.cvtColor(side_by_side[t], cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()
    try:
        from IPython.display import Image as IPImage, display
        if use_gif:
            with open(save_path, "rb") as f:
                display(IPImage(data=f.read()))
        else:
            from IPython.display import Video
            display(Video(save_path, embed=True))
    except ImportError:
        print(f"Saved to {save_path}. Open it to play.")


def plot_sequence_reconstruction(orig_seq, recon_seq, frame_idx=0):
    """
    Side-by-side 3D plot of original vs reconstructed hand at one frame.
    orig_seq, recon_seq: (T, 63) or (T, 21, 3).
    """
    orig_seq = np.asarray(orig_seq)
    recon_seq = np.asarray(recon_seq)
    if orig_seq.ndim == 2 and orig_seq.shape[1] == 63:
        orig_seq = orig_seq.reshape(-1, 21, 3)
    if recon_seq.ndim == 2 and recon_seq.shape[1] == 63:
        recon_seq = recon_seq.reshape(-1, 21, 3)
    frame_idx = min(frame_idx, len(orig_seq) - 1, len(recon_seq) - 1)
    plot_3d_hand_landmarks_pair(
        orig_seq[frame_idx],
        recon_seq[frame_idx],
        title=f"Frame {frame_idx}: Original vs Reconstructed",
    )


def plot_3d_hand_landmarks_tiled(originals, reconstructed, n_show=4):
    """
    Interactive 3D tiled layout: row 1 = original hands, row 2 = corresponding reconstructed hands.
    originals: (N, 21, 3) or (N, 63) — at least n_show samples.
    reconstructed: same shape. Each column is one sample; row 1 = original, row 2 = recon.
    """
    raise NotImplementedError(
        "Plotly interactive 3D hand plotting was disabled for this workshop cleanup. "
        "Use plot_hand_keypoints_on_image (2D overlay) instead."
    )
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("plot_3d_hand_landmarks_tiled requires plotly. Install with: pip install plotly")

    def to_21_3(a):
        a = np.asarray(a)
        if a.size == 63:
            return a.reshape(21, 3)
        return a.reshape(-1, 21, 3) if a.ndim == 2 else a

    orig = np.asarray(originals)
    recon = np.asarray(reconstructed)
    if orig.ndim == 2 and orig.shape[-1] == 63:
        orig = np.stack([to_21_3(orig[i]) for i in range(min(n_show, len(orig)))])
    elif orig.ndim == 2:
        orig = orig[:n_show].reshape(-1, 21, 3)
    else:
        orig = orig[:n_show]
    if recon.ndim == 2 and recon.shape[-1] == 63:
        recon = np.stack([to_21_3(recon[i]) for i in range(min(n_show, len(recon)))])
    elif recon.ndim == 2:
        recon = recon[:n_show].reshape(-1, 21, 3)
    else:
        recon = recon[:n_show]
    n_show = min(n_show, len(orig), len(recon))

    specs = [[{"type": "scene"}] * n_show, [{"type": "scene"}] * n_show]
    subplot_titles = [f"Original {i + 1}" for i in range(n_show)] + [f"Reconstructed {i + 1}" for i in range(n_show)]
    fig = make_subplots(
        rows=2,
        cols=n_show,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.02,
    )

    def add_hand(fig, pts, row, col, color="blue"):
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers+text",
                marker=dict(size=4, color=color),
                text=[str(i) for i in range(21)],
                textposition="top center",
            ),
            row=row, col=col,
        )
        for (i, j) in HAND_CONNECTIONS:
            fig.add_trace(
                go.Scatter3d(
                    x=[x[i], x[j]], y=[y[i], y[j]], z=[z[i], z[j]],
                    mode="lines",
                    line=dict(color=color, width=2),
                ),
                row=row, col=col,
            )

    for i in range(n_show):
        add_hand(fig, orig[i], row=1, col=i + 1, color="blue")
    for i in range(n_show):
        add_hand(fig, recon[i], row=2, col=i + 1, color="orange")

    fig.update_layout(
        title_text="Keypoint AE: Original (top row) vs Reconstructed (bottom row)",
        height=500 * 2,
    )
    fig.update_scenes(
        xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="auto",
    )
    fig.show()


def draw_hand_landmarks_on_image(rgb_image, landmarks, color_connections=(255, 100, 100), color_points=(0, 200, 255), radius=3, thickness=2):
    """
    Draw 21 hand landmarks and skeleton on an image. Modifies a copy and returns it.
    rgb_image: (H, W, 3) uint8, or (1, H, W) / (H, W) tensor/numpy (will be converted to RGB).
    landmarks: (21, 3) or (63,) — x, y are normalized in [0, 1]; z ignored for 2D draw.
    Requires cv2. Colors in BGR for cv2: (B, G, R).
    """
    if cv2 is None:
        raise ImportError("draw_hand_landmarks_on_image requires opencv-python. Install with: pip install opencv-python")
    landmarks = np.asarray(landmarks)
    if landmarks.size == 63:
        landmarks = landmarks.reshape(21, 3)
    img = np.asarray(rgb_image)
    if torch.is_tensor(img):
        img = img.cpu().numpy()
    if img.ndim == 3 and img.shape[0] == 1:
        img = img.squeeze(0)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if img.dtype != np.uint8:
        if img.max() <= 1.0 and img.min() >= -1.0:
            img = ((img + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    img = np.ascontiguousarray(img)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    H, W = img.shape[0], img.shape[1]
    for (i, j) in HAND_CONNECTIONS:
        pt1 = (int(landmarks[i, 0] * W), int(landmarks[i, 1] * H))
        pt2 = (int(landmarks[j, 0] * W), int(landmarks[j, 1] * H))
        cv2.line(img, pt1, pt2, color_connections, thickness)
    for i in range(21):
        pt = (int(landmarks[i, 0] * W), int(landmarks[i, 1] * H))
        cv2.circle(img, pt, radius, color_points, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def plot_hand_keypoints_on_image(image, kp_original, kp_reconstructed, title_left="Original", title_right="Reconstructed", figsize=(10, 5)):
    """
    Draw hand keypoints on the same image (two copies): left = raw image + original keypoints,
    right = raw image + reconstructed keypoints. Uses the same style as MediaPipe overlay
    (skeleton + points). Displays with matplotlib. Requires opencv-python.
    image: (1, H, W) or (H, W) tensor/array, normalized in [-1, 1].
    kp_original, kp_reconstructed: (21, 3) or (63,) each.
    """
    orig = np.asarray(kp_original).reshape(21, 3) if np.asarray(kp_original).size == 63 else np.asarray(kp_original)
    recon = np.asarray(kp_reconstructed).reshape(21, 3) if np.asarray(kp_reconstructed).size == 63 else np.asarray(kp_reconstructed)
    img_np = np.asarray(image)
    if torch.is_tensor(img_np):
        img_np = img_np.cpu().numpy()
    if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
        img_np = np.transpose(img_np, (1, 2, 0)) if img_np.shape[0] == 3 else img_np.squeeze(0)
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    if img_np.shape[-1] == 1:
        img_np = np.repeat(img_np, 3, axis=-1)
    if img_np.dtype != np.uint8:
        img_np = ((np.clip(img_np, -1, 1) + 1.0) * 0.5 * 255).astype(np.uint8)
    img_orig = draw_hand_landmarks_on_image(
        img_np.copy(), orig, color_connections=(255, 100, 100), color_points=(0, 200, 255)
    )
    img_recon = draw_hand_landmarks_on_image(
        img_np.copy(), recon, color_connections=(255, 180, 100), color_points=(255, 165, 0)
    )
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(img_orig)
    axes[0].set_title(title_left)
    axes[0].axis("off")
    axes[1].imshow(img_recon)
    axes[1].set_title(title_right)
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


def plot_3d_hand_landmarks_side_by_side(original, reconstructed, title_left="Original", title_right="Reconstructed"):
    """
    Two 3D hand plots side by side: left = original keypoints, right = reconstructed keypoints.
    original, reconstructed: (21, 3) or (63,) each. Requires plotly.
    """
    raise NotImplementedError(
        "Plotly interactive 3D hand plotting was disabled for this workshop cleanup. "
        "Use plot_hand_keypoints_on_image (2D overlay) instead."
    )
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("plot_3d_hand_landmarks_side_by_side requires plotly. Install with: pip install plotly")

    def to_21_3(a):
        a = np.asarray(a)
        return a.reshape(21, 3) if a.size == 63 else a

    orig = to_21_3(original)
    recon = to_21_3(reconstructed)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(title_left, title_right),
        horizontal_spacing=0.08,
    )

    def add_hand_3d(fig, pts, col, color="blue"):
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers+text",
                marker=dict(size=5, color=color),
                text=[str(i) for i in range(21)],
                textposition="top center",
            ),
            row=1, col=col,
        )
        for (i, j) in HAND_CONNECTIONS:
            fig.add_trace(
                go.Scatter3d(
                    x=[x[i], x[j]], y=[y[i], y[j]], z=[z[i], z[j]],
                    mode="lines",
                    line=dict(color=color, width=2),
                ),
                row=1, col=col,
            )

    add_hand_3d(fig, orig, col=1, color="blue")
    add_hand_3d(fig, recon, col=2, color="orange")
    fig.update_layout(height=500)
    fig.update_scenes(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="auto")
    fig.show()


def scatter_latent(latent, labels, use_tsne=True, figsize=(10, 10), title=None):
    """
    Scatter plot of latent representations colored by ground-truth labels.
    If latent dim > 2, applies t-SNE to 2D.
    latent: (N, D), labels: (N,)
    """
    latent = np.asarray(latent)
    labels = np.asarray(labels)
    if latent.shape[1] > 2 and use_tsne:
        latent = TSNE(n_components=2, random_state=42).fit_transform(latent)
    n_classes = len(np.unique(labels))
    cmap = plt.cm.get_cmap("jet", max(n_classes, 2))
    plt.figure(figsize=figsize)
    plt.scatter(
        latent[:, 0], latent[:, 1],
        c=labels, cmap=cmap, edgecolors="black", alpha=0.7
    )
    plt.colorbar(label="Class")
    plt.grid(True)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_reconstructions(original, reconstructed, n=8, figsize=(12, 4), img_shape=None):
    """
    Plot n original and reconstructed images side by side.
    original, reconstructed: (N, D) flattened or (N, 1, H, W). If flattened, pass img_shape=(H, W) e.g. (28, 28) or (128, 128).
    """
    if original.dim() == 2:
        d = original.size(1)
        if img_shape is None:
            img_shape = (28, 28) if d == 784 else (int(d**0.5), int(d**0.5))
        original = original.view(-1, 1, img_shape[0], img_shape[1])
    if reconstructed.dim() == 2:
        d = reconstructed.size(1)
        if img_shape is None:
            img_shape = (28, 28) if d == 784 else (int(d**0.5), int(d**0.5))
        reconstructed = reconstructed.view(-1, 1, img_shape[0], img_shape[1])
    n = min(n, original.size(0))
    fig, axes = plt.subplots(2, n, figsize=figsize)
    for i in range(n):
        axes[0, i].imshow(original[i].squeeze().cpu().numpy(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstructed[i].squeeze().cpu().numpy(), cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Recon")
    plt.tight_layout()
    plt.show()
