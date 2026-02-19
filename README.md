# 🧠 Sign Language Recognition using CNN, CNN-LSTM & Skeleton Keypoints

This repository presents a complete **Sign Language Recognition (SLR)** pipeline using video data.  
The project explores both **traditional machine learning** and **deep learning** approaches for recognizing sign language gestures from videos.

---

## 📌 Project Description

The system processes sign language videos and applies multiple techniques:
- Frame extraction from videos
- Skeleton keypoint extraction using MediaPipe
- Classification using Random Forest and SVM
- Deep learning using CNN and CNN-LSTM architectures
- Model evaluation and visualization
- Model conversion to TensorFlow Lite for deployment

The project supports **multi-signer data** and focuses on **20 sign classes**.

---

## 🔄 Pipeline Overview

1. **Video Preprocessing**
   - Extract frames from videos
   - Resize frames to `224 × 224`
   - Store frames by class label and signer

2. **Skeleton Keypoint Extraction**
   - Uses MediaPipe Holistic
   - Extracts left & right hand landmarks
   - Saves `(x, y)` coordinates in CSV files

3. **Data Preparation**
   - Load keypoints CSV files
   - Encode labels using `LabelEncoder`
   - Split data into training, validation, and testing sets

---

## 🤖 Models Implemented

### 🔹 Traditional Machine Learning
- **Random Forest Classifier**
  - Hyperparameter tuning using Grid Search
- **Support Vector Machine (SVM)**
  - Kernel and regularization tuning

**Evaluation Metrics**
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

### 🔹 Deep Learning Models

#### 🧠 CNN Model
- Input: RGB images
- Convolution + Batch Normalization + MaxPooling
- Dropout for regularization
- Softmax classification layer

#### 🧠 CNN-LSTM Model
- CNN for spatial feature extraction
- LSTM layers for temporal modeling
- Designed for video sequence learning
- Early stopping and TensorBoard logging

---

## 📊 Evaluation & Visualization

- Confusion matrices using `ConfusionMatrixDisplay`
- Metrics: Accuracy, Precision, Recall, F1-score, AUC
- TensorBoard for training monitoring

---

## 💾 Model Saving & Deployment

- Models saved in `.h5` format
- Converted to **TensorFlow Lite (`.tflite`)**
- Suitable for mobile and edge-device deployment


## 🛠️ Technologies Used

- Python 3.9
- OpenCV
- MediaPipe
- NumPy & Pandas
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- TensorBoard

---

## 🚀 How to Run

1. Install dependencies:
```bash
pip install opencv-python mediapipe tensorflow scikit-learn pandas numpy matplotlib tqdm
