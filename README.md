# VQ-VAE Workshop

Tutorial for using VQ-VAE–style models for **unsupervised clustering**, stepwise from autoencoders to vector quantization.

## Setup

```bash
pip install -r requirements.txt
```

## How to run

1. Open the Jupyter notebooks in order:
   - **01_mnist_linear_ae.ipynb** — Linear AE on MNIST, latent clustering.
   - **02_hand_image_keypoint_ae.ipynb** — Hand keypoint (63-dim) AE (trained on keypoints; images used only for visualization overlays).
   - **03_cnn_ae.ipynb** — CNN AE on MNIST.
   - **04_gru_ae.ipynb** — GRU AE on 20-frame keypoint sequences.
   - **05_vq_mnist.ipynb** — VQ-VAE on MNIST, clustering by codebook index.
2. Run cells top to bottom; data will download automatically where applicable (e.g. MNIST).
3. Hand/keypoint and sequence data: place in `data/` or set the path as documented in each notebook; demos use synthetic data if files are missing.

## Plan

See [PLAN.md](PLAN.md) for the full roadmap (datasets, models, phases).
