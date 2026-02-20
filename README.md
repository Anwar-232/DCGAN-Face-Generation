# DCGAN-Face-Generation
# Human Face Generation using DCGAN

This repository contains a PyTorch implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) designed to synthesize realistic human faces from random noise.

## 📌 Overview
Generative Adversarial Networks (GANs) consist of two competing neural networks:
- **The Generator:** Learns the underlying statistical distribution of the training data to create hyper-realistic fake images.
- **The Discriminator:** Acts as an artificial detective, learning to distinguish between real images from the dataset and fake images produced by the generator.
Through adversarial training, both networks improve until the generator produces images that are indistinguishable from reality.

## 📊 Dataset: CelebA
The model is trained on the [CelebFaces Attributes (CelebA) Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset), which contains over 200,000 celebrity images.
- **Preprocessing:** Images are cropped, resized to `64x64` pixels, and normalized to the `[0, 1]` range using standard PyTorch transformations.

## 🏗️ Model Architecture
The architecture strictly follows the DCGAN guidelines to ensure stable training:
1. **Generator (G):** Utilizes sequential `ConvTranspose2d` layers paired with `BatchNorm2d` and `ReLU` activations. It takes a 100-dimensional latent noise vector (`z_dim=100`) and upsamples it to a `3x64x64` RGB image using a `Sigmoid` activation output.
2. **Discriminator (D):** Built with sequential `Conv2d` layers, `BatchNorm2d`, and `LeakyReLU (0.2)` activations to classify images as real or fake, outputting a probability score via a `Sigmoid` function.

## 🚀 Training Details
- **Framework:** PyTorch
- **Loss Function:** Binary Cross-Entropy (BCE Loss)
- **Optimizer:** Adam (Learning Rate: `2e-4`, Betas: `0.5, 0.999`)
- **Epochs:** 15
- **Hardware:** Trained on a T4 GPU.

## 📈 Results & Analysis
The adversarial training demonstrated a healthy balance between the two networks:
- **Loss Curves:** The Discriminator's loss gradually stabilized around `0.3`, while the Generator's loss fluctuated between `3.0` and `5.0`. There were no signs of mode collapse, ensuring diverse facial generations.
- **Visual Evolution:** - *Epochs 1-3:* The network outputted blurry, abstract facial structures.
  - *Epochs 7-10:* Key facial features (eyes, hair, skin tone) became coherent.
  - *Epoch 15:* The model successfully generated high-fidelity faces with consistent lighting and spatial proportions.

## 🔮 Future Improvements
- Extend training to 30-50 epochs for finer details.
- Shift data normalization to the `[-1, 1]` range with a `Tanh` output layer in the generator for enhanced mathematical stability.
- Explore advanced architectures like **WGAN-GP** or **StyleGAN** for higher resolution outputs.

## 🛠️ Usage
1. Clone the repository.
2. Download the CelebA dataset from Kaggle and place it in the designated directory.
3. Run the notebook `CelebA_Face_Generation_DCGAN.ipynb` to initiate training.
4. Generated samples and model weights are periodically saved in the `gan_outputs/` directory.
