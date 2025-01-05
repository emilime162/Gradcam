# Grad-CAM and Deconvolutional Visualization with PyTorch

PyTorch implementation of **Grad-CAM (Gradient-weighted Class Activation Mapping)** and **Deconvolutional Visualization** techniques for understanding image classification models. This project demonstrates how to visualize the regions of an image that influence the model's predictions and provides insights into adversarial robustness.

---

## Features

* Visualizes **Grad-CAM heatmaps** for specific layers of a pretrained VGG16 model.
* Supports **Deconvolutional Visualization** to reconstruct input features from activations.
* Generates a **2x4 grid** combining the input image and Grad-CAM visualizations.
* Analyzes and visualizes adversarial examples to understand model vulnerabilities.

---

## Requirements

* Python 3.6 or higher
* PyTorch
* torchvision
* numpy
* opencv-python
* matplotlib
* click

Install the required dependencies with:

```bash
pip install torch torchvision numpy opencv-python matplotlib click


