<div align="center">

# 🧠 Handwritten Digit Recognition & OCR System

> *From raw pixels to recognized text — powered by ML & Deep Learning*

<br>

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-TrOCR-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Status](https://img.shields.io/badge/Status-Completed-22C55E?style=for-the-badge)

<br>

**KNN · HOG+SVM · TrOCR · EasyOCR · MNIST**

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Workflow](#-workflow)
- [Models](#-models-used)
- [Dataset](#-dataset)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Tech Stack](#-tech-stack)
- [Use Cases](#-use-cases)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🚀 Overview

A complete end-to-end **Optical Character Recognition (OCR)** pipeline for handwritten digits, combining classical Machine Learning and state-of-the-art Deep Learning in a unified system.

The project progresses from a simple KNN baseline all the way to a **Transformer-based TrOCR model**, demonstrating how each layer of complexity improves real-world recognition performance.

| Layer | Technology | Purpose |
|---|---|---|
| 📸 Vision | OpenCV | Image preprocessing & segmentation |
| 📊 ML | KNN, HOG + SVM | Classical digit recognition |
| 🤖 DL | TrOCR, EasyOCR | Transformer-powered real-world OCR |

---

## ✨ Features

```
✅  Handwritten digit recognition on MNIST & real-world images
✅  Noise removal, grayscale conversion & Otsu thresholding
✅  Contour-based character segmentation
✅  Multi-model comparison: KNN vs SVM vs TrOCR
✅  HOG feature extraction for robust structural representation
✅  Prediction visualization with bounding boxes
✅  Google Colab–ready with image upload support
```

---

## 🏗️ Workflow

```
┌─────────────────────────────────────────────────────┐
│                    INPUT IMAGE                      │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│           PREPROCESSING                             │
│   • Grayscale conversion                            │
│   • Gaussian blur (noise reduction)                 │
│   • Otsu's thresholding (binarization)              │
│   • Morphological operations                        │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│           SEGMENTATION                              │
│   • Contour detection                               │
│   • Bounding box extraction                         │
│   • Character isolation                             │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│           FEATURE EXTRACTION                        │
│   • Raw pixels (784 features) for KNN               │
│   • HOG descriptors for SVM                         │
│   • Patch embeddings for TrOCR                      │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌──────────────┬──────────────────┬────────────────────┐
│     KNN      │    HOG + SVM     │   TrOCR / EasyOCR  │
└──────┬───────┴────────┬─────────┴──────────┬──────────┘
       │                │                    │
       └────────────────┴────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                 OUTPUT TEXT / DIGIT                 │
└─────────────────────────────────────────────────────┘
```

---

## 🧪 Models Used

### 🔹 KNN — Baseline Classifier
> *Simple, interpretable, surprisingly effective*

- Uses raw 28×28 pixel values flattened to **784 features**
- Distance-based classification (Euclidean)
- Great for understanding the fundamentals

### 🔸 HOG + SVM — Feature Engineering Approach
> *Structural features meet a powerful boundary-finding classifier*

- **Histogram of Oriented Gradients (HOG)** captures edge and shape information
- SVM finds the optimal decision hyperplane
- More robust to variations in stroke width and positioning

### 🔺 TrOCR + EasyOCR — Transformer-Powered OCR
> *Industry-grade recognition for real-world handwriting*

- **EasyOCR** handles scene text detection
- **TrOCR** (Microsoft) uses a Vision Encoder + Language Decoder
- Pre-trained on millions of handwritten samples
- Works directly on photographs and scanned documents

---

## 📂 Dataset

<div align="center">

### MNIST Handwritten Digits

| Split | Samples | Image Size | Classes |
|-------|---------|------------|---------|
| Training | 60,000 | 28 × 28 px | 10 (0–9) |
| Testing | 10,000 | 28 × 28 px | 10 (0–9) |

</div>

```python
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train shape: (60000, 28, 28) — grayscale pixel arrays
```

---

## 📊 Results

<div align="center">

| Model | Accuracy | Speed | Real-World Ready |
|---|---|---|---|
| 🔹 KNN | ~95% | ⚡ Fast | ❌ |
| 🔸 HOG + SVM | ~97–98% | ⚡ Fast | ⚠️ Limited |
| 🔺 TrOCR | ✨ High | 🐢 Moderate | ✅ Yes |
| 🌀 EasyOCR | ✨ High | 🐢 Moderate | ✅ Yes |

</div>

> **Takeaway:** KNN and SVM work great for clean MNIST-style digits. TrOCR is the go-to when working with real photographs or complex handwriting.

---

## ⚙️ Installation

```bash
# Core ML & vision libraries
pip install opencv-python scikit-learn matplotlib numpy

# Deep learning & OCR
pip install transformers easyocr pillow torch torchvision

# Dataset loading
pip install keras tensorflow
```

---

## ▶️ Usage

### Step 1 — Train the Models

Open the notebook and run cells to train:
- KNN on MNIST pixel features
- SVM on HOG-extracted features

### Step 2 — Upload a Handwritten Image *(Google Colab)*

```python
from google.colab import files
uploaded = files.upload()  # Select your handwritten image
```

### Step 3 — Run the Full OCR Pipeline

```python
# Runs preprocessing → segmentation → prediction
run_ocr_on_image("your_image.png")
```

### Step 4 — View Predictions

```python
# Displays bounding boxes and predicted digits on the image
visualize_predictions("your_image.png", predictions)
```

---

## 🛠️ Tech Stack

<div align="center">

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| Vision | OpenCV |
| ML | Scikit-learn |
| DL Framework | PyTorch |
| OCR | EasyOCR, HuggingFace TrOCR |
| Data & Viz | NumPy, Matplotlib |
| Environment | Google Colab / Jupyter |

</div>

---

## 💡 Use Cases

| Domain | Application |
|---|---|
| 📦 Logistics | Postal/ZIP code scanning |
| 🧾 Finance | Cheque amount reading |
| 🏫 Education | Automated answer sheet grading |
| 🏥 Healthcare | Handwritten prescription digitization |
| 🤖 Automation | Any OCR pipeline requiring digit extraction |

---

## 🔮 Future Improvements

- [ ] 🧠 Add a **custom CNN classifier** trained from scratch
- [ ] 🌐 Deploy as a **Streamlit web app** with drag-and-drop upload
- [ ] ⚡ Expose a **FastAPI REST endpoint** for programmatic access
- [ ] 📄 Support **multi-line text recognition**
- [ ] 🔤 Extend to **full alphanumeric OCR** (A–Z, 0–9)
- [ ] 📱 Package as a **mobile-friendly inference API**

---

## 👨‍💻 Author

<div align="center">

**Shylesh S**

*AI & Data Science Student*

Machine Learning · Generative AI · Automation

<br>

[![Email](https://img.shields.io/badge/Email-shylesh1640%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:shylesh1640@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](#)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](#)

</div>

---

<div align="center">

### ⭐ Found this useful? Star the repo and share it!

*Built with ❤️ using Python, OpenCV, and HuggingFace Transformers*

</div>
