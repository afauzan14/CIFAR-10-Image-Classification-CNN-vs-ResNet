# Comparative Image Classification on CIFAR-10: Custom CNN vs. ResNet-18
### Benchmarking Architectural Depth, Residual Learning, and Feature Representation in PyTorch

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-2496ED?logo=pytorch&logoColor=white)](https://pytorch.org/vision/)
[![Jupyter Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter&logoColor=white)](CNN.ipynb)
[![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-blue)](https://www.cs.toronto.edu/~kriz/cifar.html)
[![Top Accuracy](https://img.shields.io/badge/Top%20Accuracy-94.64%25-success)]()

---

## Project Overview

This project presents an empirical computer vision study benchmarking a **Custom Convolutional Neural Network (CNN)** against a **Deep Residual Network (ResNet-18)** for multi-class object classification on the **CIFAR-10** benchmark.

The core objectives of this investigation include:
1. **Architectural Comparison:** Evaluating the trade-offs between a lightweight sequential CNN (8 convolutional layers) and a deep residual architecture utilizing identity skip-connections.
2. **Optimization Dynamics:** Analyzing the convergence effects of different optimization strategies (Adam with `ReduceLROnPlateau` versus SGD with `CosineAnnealingLR`).
3. **Interpretability & Feature Map Extraction:** Visualizing intermediate activation layers to verify hierarchical feature extraction from low-level edges to semantic object structures.

---

## Dataset and Augmentation Pipeline

* **Dataset:** CIFAR-10 (60,000 $32 \times 32$ color images partitioned into 10 distinct object categories: *airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck*).
* **Data Splits:** 45,000 Training images, 5,000 Validation images, and 10,000 Test images (seeded with `seed=42` for strict reproducibility).
* **Data Augmentation:**
  * Random Horizontal Flip ($p = 0.5$)
  * Random Cropping with padding ($4$ pixels)
  * Tensor Normalization using channel-wise CIFAR-10 mean (`[0.4914, 0.4822, 0.4465]`) and standard deviation (`[0.2470, 0.2435, 0.2616]`).

---

## Model Architectures and Hyperparameters

Both architectures were implemented natively using **PyTorch**:

| Hyperparameter / Component | Custom CNN Configuration | ResNet-18 Configuration |
| :--- | :--- | :--- |
| **Architectural Depth** | Shallow (8 Conv Layers + MaxPool + BatchNorm) | Deep (18 Layers with Residual Skip-Connections) |
| **Optimizer** | Adam | SGD (Momentum = 0.9, Weight Decay = $5 \times 10^{-4}$) |
| **Initial Learning Rate** | $0.001$ | $0.1$ |
| **Batch Size** | 32 | 128 |
| **LR Scheduler** | `ReduceLROnPlateau` (factor = 0.1, patience = 3) | `CosineAnnealingLR` ($T_{\max} = 100$) |
| **Loss Function** | Cross-Entropy Loss | Cross-Entropy Loss |

---

## Comparative Performance Evaluation

Evaluated across the 10,000 unseen test images:

| Architecture | Training Accuracy | Validation Accuracy | Test Accuracy | Generalization Gap |
| :--- | :---: | :---: | :---: | :---: |
| **Custom CNN** | 89.59% | 86.84% | **87.25%** | 2.34% (Stable) |
| **ResNet-18** | 99.97% | 95.26% | **94.64%** | 5.33% (Superior) |

### Key Observations
* **Performance Gain:** ResNet-18 achieved a **+7.39% accuracy improvement** over the Custom CNN on the holdout test set.
* **Residual Learning Impact:** Identity skip-connections effectively mitigated the vanishing gradient problem, allowing the 18-layer network to converge smoothly and capture fine-grained visual features.
* **Scheduler Synergy:** The pairing of **SGD + CosineAnnealingLR** proved significantly more effective for deep convergence than Adam on this vision benchmark.

---

## Feature Map Analysis and Model Interpretability

* **Early Convolutional Layers:** Both architectures learn fundamental spatial features (oriented edges, textures, and color gradients).
* **Deeper Layers:** ResNet-18 activation maps preserve significantly sharper localized activations and high-level class-specific representations, whereas the shallow CNN activations exhibit spatial diffusion.
* **Class Confusion Dynamics:** Misclassifications were predominantly concentrated among semantically similar animal classes (e.g., *cat* vs. *dog*) and vehicle categories (e.g., *automobile* vs. *truck*), with ResNet displaying substantially lower confusion rates across challenging classes.

---

## Summary of Findings

1. **Residual Connections Enable Depth:** While standard sequential convolutions plateau in accuracy, residual connections enable deeper architectures without suffering from optimization degradation.
2. **Optimization Strategy Matters:** Modern learning rate schedules (`CosineAnnealingLR`) combined with momentum-based SGD provide stronger generalization than standard adaptive optimizers in deep vision tasks.
3. **Deployment Trade-Off:** The Custom CNN (87.25%) offers an ultra-lightweight alternative suitable for low-compute edge devices, while ResNet-18 (94.64%) delivers production-grade classification accuracy.
