<img width="827" alt="Screenshot 2025-03-06 at 12 32 35 AM" src="https://github.com/user-attachments/assets/7340f2c7-7b9d-48c6-9ed7-77b3ffa7846b" />

# Politician Face Classification Using Deep Learning

# Face Recognition Model

## 📌 Project Overview
This project implements a **Face Recognition System** using **PyTorch** and a **pretrained ResNet-18 model**. It performs **face classification** by training on cropped face images and evaluates performance using various metrics such as **accuracy, precision, recall, and F1-score**.

## 🚀 Features
- **Pretrained ResNet-18** for feature extraction.
- **Data Augmentation** for robust training.
- **Train/Test Split (80/20)** for evaluation.
- **Dynamic Learning Rate Adjustment** with `ReduceLROnPlateau`.
- **Early Stopping** for efficiency.
- **Model Saving & Loading** for future inference.
- **Confusion Matrix & Performance Metrics** visualization.

## 🖥️ Installation
```bash
# Clone the repository
git clone git@github.com:your-username/image-classification.git
cd image-classification

# Install dependencies
pip install -r requirements.txt
```

## 📂 Dataset Structure
Make sure your dataset follows this structure:
```
dataset/cropped/
│── Donald Trump/
│   ├── image1.jpg
│   ├── image2.jpg
│── Elon Musk/
│   ├── image1.jpg
│   ├── image2.jpg
│── Joe Biden/
│   ├── image1.jpg
│   ├── image2.jpg
│── Vladimir Putin/
│   ├── image1.jpg
│   ├── image2.jpg
```

## 📊 Model Training
Run the training script:
```bash
python train.py
```

### **Key Training Parameters**
- **Batch Size**: `8`
- **Learning Rates**:
  - `fc: 0.001` (highest for classification layer)
  - `layer4: 0.0005`, `layer3: 0.0003`, `layer2: 0.0001`
  - `layer1: 0.00005`, `conv1: 0.00001`
- **Epochs**: `25`
- **Early Stopping**: Enabled after `5` non-improving epochs

## 📈 Model Evaluation
The model automatically evaluates performance at the end of training. Metrics include:
- **Final Validation Accuracy**
- **Per-Class Accuracy**
- **Precision, Recall, and F1 Score**

## 🎯 Inference: Predict a Face
Use the trained model to predict a new face:
```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def predict_image(image_path, model, transform):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        probability = torch.nn.functional.softmax(output, dim=1)[0]
    
    return predicted.item(), probability.cpu().numpy()

# Load Model
checkpoint = torch.load('face_recognition_model.pth')
model = models.resnet18()
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.4),
    torch.nn.Linear(512, checkpoint['num_classes'])
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Transform for Inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Predict
class_id, probs = predict_image('path/to/new_face.jpg', model, transform)
class_names = checkpoint['class_names']
predicted_person = class_names[class_id]
confidence = probs[class_id]
print(f'Predicted: {predicted_person} with confidence: {confidence:.4f}')
```

## 📊 Visualization
The training script saves:
1. **Confusion Matrix** (`confusion_matrix.png`)
2. **Loss & Accuracy Curves** (`training_curves.png`)

## 💾 Model Checkpoints
The best model is saved as:
```
best_model.pth
face_recognition_model.pth
```

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first.

## 📜 License
MIT License. See `LICENSE` for details.
