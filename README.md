# DermaMNIST Deep Learning Classification
> EPFL CS-233 – Introduction to Machine Learning

Implementation of **MLP** and **CNN** deep networks in PyTorch to classify
dermatoscopic images from the **DermaMNIST** dataset (7 diagnostic categories).

---

## 1. Introduction
This project compares a fully-connected **MLP** and a **CNN** architecture
for the 7-class DermaMNIST skin-lesion dataset (9,012 images, 28×28 RGB).
Goal: predict the correct diagnostic category for each image.

---

## 2. Implementation

### Architectures
- **MLP**  
  Three hidden layers (512 → 256 → 128) with ReLU, BatchNorm and Dropout(0.6).  
  Input images are flattened to vectors.
- **CNN**  
  Three convolutional blocks (32 → 64 → 128 channels) with 3×3 kernels,
  ReLU, BatchNorm, MaxPooling and Dropout(0.7), followed by two fully-connected layers.

### Training
- Optimizer: **Adam** with learning-rate scheduling (`ReduceLROnPlateau`)
- Learning rate: `5e-5`
- Batch size: `32`
- Epochs: `30`
- Dropout: `0.6` (MLP), `0.7` (CNN)
- Weight decay (L2 regularization): `1e-3`
- Early stopping to prevent overfitting
- Automatic **CPU/GPU** selection (CUDA if available)

---

## 3. Results

| Model | Train Accuracy | Validation Accuracy |
|------ |---------------:|--------------------:|
| **MLP** | **73.2 %** | 70.7 % |
| **CNN** | 71.4 % | **70.9 %** |

- **Overfitting mitigation**: increased dropout, lower learning rate,
  smaller batch size, L2 regularization.
- CNN showed slightly better **generalization** and a more structured
  confusion matrix, while MLP achieved higher training accuracy.

---

## 4. Features
- Training logs (loss, accuracy, runtime)
- Learning rate scheduling
- Dynamic device support (CPU/CUDA)

---

## 5. Repository Content
This repository contains **only my own implementation files**  
(`main.py`, `src/methods/deep_network.py`, training utilities).  

---
## 6. How to Run
```bash ```
## Clone the repository
```bash
git clone https://github.com/<linakbj>/dermamnist-deep-learning.git
cd dermamnist-deep-learning
```

## 2. Download the DermaMNIST dataset
Get it from https://medmnist.com/
and place the extracted folder in:
dermamnist-deep-learning/data/dermamnist/

## 3. Install dependencies
```bash
pip install numpy torch matplotlib opencv-python==4.6.0.66
```

## 4. Train or evaluate a model
##    Train a Convolutional Neural Network (CNN)
```bash
python main.py --data data/dermamnist --nn_type cnn --lr 5e-5 --max_iters 30
```

##    Train a Multi-Layer Perceptron (MLP)
```bash
python main.py --data data/dermamnist --nn_type mlp --lr 5e-5 --max_iters 30
```
