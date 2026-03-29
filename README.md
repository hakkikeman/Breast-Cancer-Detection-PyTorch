# Breast Cancer (IDC) Detection using Deep Learning

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hakkikeman/Breast-Cancer-Detection-PyTorch/blob/main/breast_cancer_classification_cnn_vs_vgg16.ipynb)

## Project Overview

This project focuses on the binary classification of Invasive Ductal Carcinoma (IDC), the most common type of breast cancer, using histopathological images. The study compares the performance of a Custom Convolutional Neural Network (CNN) built from scratch with a pre-trained VGG-16 transfer learning model.

## Dataset

The dataset used is the "Breast Histopathology Images" from Kaggle. A balanced subset of 10,000 RGB image patches ($50\times50$ pixels) was utilized. The data was split into 70% Training, 15% Validation, and 15% Testing. *(Note: The dataset is not included in this repository due to size constraints. You can download the [Breast Histopathology Images Dataset](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) directly from Kaggle).*

## Methodology & Models

The entire workflow is implemented in PyTorch within a single Jupyter Notebook.

* **Custom CNN:** Designed from scratch with 3 Convolutional blocks, Max Pooling, ReLU activation, 50% Dropout, and Fully Connected layers.
* **VGG-16 (Transfer Learning):** Utilized pre-trained ImageNet weights. The final classifier layer was fine-tuned for this specific medical imaging task.
* **Generative AI Integration:** Generative AI (Gemini) was utilized to research and implement advanced optimization strategies, specifically applying "Differential Learning Rates" to further enhance the VGG-16 model's performance during fine-tuning.

## Results

Both models were trained on an NVIDIA T4 GPU (Google Colab).

* **Custom CNN:** Achieved **85.13%** Validation Accuracy.
* **VGG-16:** Achieved **83.60%** Validation Accuracy.
  The Custom CNN adapted more quickly to the specific tissue details within the limited epoch count (5 epochs).

## How to Run

1. Clone this repository.
2. Download the dataset from Kaggle and update the data path in the notebook.
3. Install the required dependencies (`torch`, `torchvision`, `pandas`, `scikit-learn`, `matplotlib`).
4. Run all cells in `breast_cancer_detection_cnn_vs_vgg16.ipynb`.
