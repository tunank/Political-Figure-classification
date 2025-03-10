<img width="827" alt="Screenshot 2025-03-06 at 12 32 35 AM" src="https://github.com/user-attachments/assets/7340f2c7-7b9d-48c6-9ed7-77b3ffa7846b" />

# Face Recognition Model

## 📌 Project Overview
This project implements a **Face Recognition System** using **PyTorch** and a **pretrained ResNet-18 model**. It performs **face classification** by training on cropped face images and evaluates performance using various metrics such as **accuracy, precision, recall, and F1-score**.

## 🚀 Features
- **Pretrained ResNet-18** for feature extraction.
- **Data Augmentation** for robust training.
- **Train/Test Split (70/30)** for evaluation.
- **Dynamic Learning Rate Adjustment** with `ReduceLROnPlateau`.
- **Model Saving & Loading** for future inference.
- **Confusion Matrix & Performance Metrics** visualization.

## 📈 Model Evaluation
The model automatically evaluates performance at the end of training. Metrics include:
- **Final Validation Accuracy**
- **Per-Class Accuracy**
- **Precision, Recall, and F1 Score**

## 📊 Visualization
The training script saves:
1. **Confusion Matrix**
2. **Loss & Accuracy Curves**

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first.

## 📜 License
MIT License. See `LICENSE` for details.
