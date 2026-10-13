# Celebrity Face Generation with a VAE

A Variational Autoencoder (VAE) written from scratch in PyTorch and trained on the [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) dataset. It reconstructs faces, samples brand new ones from random noise, and lets you push a face around in latent space to see what the model changes.

| Original reconstruction | Forced latent change |
| :---: | :---: |
| <img src="results/original_face.png" width="300"> | <img src="results/forced_change.png" width="300"> |

## What it does

- **Reconstruction:** encodes a 64×64 face into a 128-dimensional latent vector and decodes it back.
- **Generation:** samples `z ~ N(0, I)` and decodes it into a face that never existed.
- **Latent editing:** adds a fixed offset to part of a face's latent vector and decodes the result, to show how the latent space behaves and how entangled it is.

## Model

| Part | Details |
| --- | --- |
| Input | 64×64 RGB image, pixel values scaled to [0, 1] |
| Encoder | 3 × `Conv2d` (kernel 4, stride 2, padding 1): 3 → 32 → 64 → 128 channels, ReLU, giving a 128×8×8 feature map |
| Latent head | Flatten (8192) → two `Linear` layers producing `mu` and `logvar` (128-dim each) |
| Sampling | Reparameterization trick: `z = mu + exp(0.5·logvar) · ε`, with `ε ~ N(0, I)` |
| Decoder | `Linear` 128 → 8192, reshape to 128×8×8, 3 × `ConvTranspose2d` (128 → 64 → 32 → 3), ReLU between layers, Sigmoid at the output |
| Loss | Summed binary cross-entropy (reconstruction) + KL divergence to the standard normal prior |

## Training

| Setting | Value |
| --- | --- |
| Data | 202,599 CelebA images, resized to 64×64 |
| Batch size | 64 (shuffled) |
| Optimizer | Adam, learning rate 1e-3 |
| Epochs | 10 |

Average per-image loss (BCE + KL) over the 10 epochs:

| Epoch | 1 | 2 | 3 | 5 | 10 |
| --- | --- | --- | --- | --- | --- |
| Loss | 6438.50 | 6290.00 | 6269.43 | 6255.85 | 6245.04 |

## Repository layout

The notebooks are meant to be run in order, in a single session (Google Colab works well), because later notebooks reuse variables such as `train_dataloader` and `vae_model` from earlier ones.

| Notebook | Purpose |
| --- | --- |
| [`datasetdownload.ipynb`](datasetdownload.ipynb) | Downloads and unzips CelebA from Kaggle |
| [`preprocessing.ipynb`](preprocessing.ipynb) | Resize to 64×64, convert to float tensors, build the `DataLoader` |
| [`encoderdecoder.ipynb`](encoderdecoder.ipynb) | Builds the encoder, reparameterization and decoder piece by piece and checks tensor shapes |
| [`intilization and training.ipynb`](intilization%20and%20training.ipynb) | Full `VAE` class, loss function and training loop |
| [`outputs.ipynb`](outputs.ipynb) | Reconstructions, random face generation, latent-space editing |
| [`results/`](results) | Images used in this README |

## How to run

1. Clone the repo:
   ```bash
   git clone https://github.com/gladiator456/Celebrity-Face-Generation-VAE.git
   cd Celebrity-Face-Generation-VAE
   ```
2. Install the dependencies:
   ```bash
   pip install torch torchvision matplotlib numpy kaggle
   ```
3. Set up Kaggle access. Create an API token in your Kaggle account settings and expose it as environment variables **outside the notebook file**, so it never gets committed:
   ```bash
   export KAGGLE_USERNAME="your-username"
   export KAGGLE_KEY="your-key"
   ```
4. Run the notebooks in this order: `datasetdownload` → `preprocessing` → `intilization and training` → `outputs`. (`encoderdecoder` is an optional walkthrough of the architecture.)

A GPU is strongly recommended. The code picks CUDA automatically when it is available.

## Latent-space editing

`outputs.ipynb` takes the encoder mean `mu` of one face, adds `+3.5` to its first 15 latent dimensions, and decodes both versions side by side. These dimensions were chosen arbitrarily, not found by analysing which directions correspond to a particular attribute. The change shows up as shading and darker regions around the eyes along with side effects such as shifts in hair and apparent gender, which is a good illustration of **latent entanglement**: individual dimensions of a plain VAE do not map cleanly onto single human-readable features.

## Limitations

- Outputs are blurry, which is typical of a VAE trained with pixel-wise BCE loss.
- Training is short (10 epochs) and the loss was still decreasing slowly when it stopped.
- Images are low resolution (64×64).
- The model weights are not saved, so the notebooks have to be re-run to reproduce results.

## Ideas for next steps

- Train longer and save checkpoints.
- Find real attribute directions using CelebA's attribute labels (for example the mean latent of "Smiling" minus "Not smiling") instead of an arbitrary offset.
- Try a β-VAE to encourage more disentangled latents.
- Add a perceptual or adversarial loss for sharper faces.

## Tech stack

Python, PyTorch, Torchvision, NumPy, Matplotlib.

## 🛠️ Tech Stack & Concepts Used
- **Language:** Python
- **Framework:** PyTorch
- **Libraries:** NumPy, Matplotlib, Torchvision
- **Key Algorithms:** Stochastic Gradient Descent, Reparameterization Trick, KL-Divergence Loss, Structural Convolutions.

## 🏃 How to Run the Project
1. Clone this repository:
   ```bash
   git clone [https://github.com/gladiator456/Celebrity-Face-Generation-VAE.git](https://github.com/gladiator456/Celebrity-Face-Generation-VAE.git)
