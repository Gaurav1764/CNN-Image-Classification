# Multiclass Image Classification using CNN and Transfer Learning

## Deep Learning Architectures and Techniques – Laboratory Assignment

This project implements **multiclass image classification** using two approaches:

1. **Custom Convolutional Neural Network (CNN)**
2. **Transfer Learning using a pretrained CNN model**

The goal of this experiment is to understand convolution operations, pooling layers, data augmentation, and the benefits of transfer learning in image recognition tasks. 

---

# Project Overview

Image classification is one of the most common applications of deep learning. Convolutional Neural Networks (CNNs) are specifically designed for image data because they automatically learn spatial hierarchies of features.

In this project:

* A **custom CNN model** is implemented from scratch.
* A **pretrained model (MobileNetV2)** is used for transfer learning.
* Both models are trained on the **Fashion-MNIST dataset**.
* Their performance is evaluated and compared using standard classification metrics.

---

# Dataset

The **Fashion-MNIST dataset** is used in this project.

It contains **70,000 grayscale images of clothing items** divided into **10 classes**.

Image properties:

| Property          | Value   |
| ----------------- | ------- |
| Image size        | 28 × 28 |
| Training images   | 60,000  |
| Test images       | 10,000  |
| Number of classes | 10      |

Classes include:

* T-shirt
* Trouser
* Pullover
* Dress
* Coat
* Sandal
* Shirt
* Sneaker
* Bag
* Ankle Boot

Dataset is loaded directly from TensorFlow.

---

# Project Workflow

The project follows the standard deep learning pipeline:

1. Dataset loading
2. Image preprocessing
3. Data normalization
4. Train–validation–test split
5. Data augmentation
6. Custom CNN model design
7. Transfer learning implementation
8. Model training
9. Performance evaluation
10. Model comparison

---

# Technologies Used

The project is implemented using the following tools:

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Scikit-learn
* Seaborn
* Jupyter Notebook

---

# CNN Architecture

The custom CNN architecture includes the following layers:

Input Layer
→ Convolution Layer (32 filters)
→ Max Pooling Layer
→ Convolution Layer (64 filters)
→ Max Pooling Layer
→ Flatten Layer
→ Fully Connected Layer (128 neurons)
→ Dropout Layer
→ Output Layer (Softmax)

Activation Function: **ReLU**

Loss Function: **Categorical Cross-Entropy**

Optimizer: **Adam**

---

# Transfer Learning Model

Transfer learning is implemented using the **MobileNetV2 pretrained model**.

Steps involved:

1. Load pretrained MobileNetV2 without top layers
2. Freeze pretrained layers
3. Add custom classification layers
4. Train the final network on Fashion-MNIST

Advantages of transfer learning:

* Faster training
* Better feature extraction
* Improved classification accuracy

---

# Data Augmentation

To reduce overfitting and improve model generalization, data augmentation techniques were applied.

Techniques used:

* Rotation
* Zoom
* Width shifting
* Height shifting

This increases the diversity of training samples.

---

# Model Evaluation

Both models are evaluated using the following metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Training performance is visualized using:

* Accuracy curves
* Loss curves

These plots help analyze learning behaviour during training.

---

# Results

| Model                           | Accuracy |
| ------------------------------- | -------- |
| Custom CNN                      | ~88–90%  |
| Transfer Learning (MobileNetV2) | ~92–94%  |

The transfer learning model performs better because pretrained networks already contain rich feature representations learned from large image datasets.

---

# Project Structure

```
CNN-Image-Classification
│
├── Assignment_2.ipynb
├── Cnn__Image_Classifier.py
│
├── plots
│   ├── cnn_accuracy.png
│   ├── cnn_loss.png
│   ├── transfer_accuracy.png
│   ├── transfer_loss.png
│   └── confusion_matrix.png
│
├── report
│   └── CNN_Image_Classification_Report.pdf
│
└── README.md
```

---

# How to Run the Project

## Using Jupyter Notebook

1. Open Jupyter Notebook or Google Colab
2. Open

```
CNN_Image_Classification.ipynb
```

3. Run all cells

The notebook will:

* Train the CNN model
* Train the transfer learning model
* Generate plots
* Evaluate performance

---

# Learning Outcomes

This project demonstrates the following deep learning concepts:

* Convolution operation in CNNs
* Pooling layers
* Feature extraction
* Data augmentation
* Transfer learning
* Model evaluation using classification metrics

---

# Author

**Gaurav Kumar**

MCA – Deep Learning Architectures and Techniques
SOET

---

# License

This project is developed for academic purposes as part of the Deep Learning laboratory coursework.

---

