# VQ-VAE Workshop

Tutorial for using VQ-VAE–style models for **unsupervised clustering**, stepwise from autoencoders to vector quantization.

## Setup

```bash
pip install -r requirements.txt
```

## How to run

1. Open the Jupyter notebooks in order:
   - **00_Setup.ipynb** — To setup the dependencies on Google Colaba and download the datasets.
   
   - **01_mnist_linear_ae.ipynb** — Linear AE on MNIST, latent clustering.
   - **02_hand_image_keypoint_ae.ipynb** — Hand keypoint (63-dim) AE (trained on keypoints; images used only for visualization overlays).
   - **03_salsa.ipynb** — Salsa Dance move sequences (19x262) AE and VQ-VAE.
   