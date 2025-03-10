<img width="827" alt="Screenshot 2025-03-06 at 12 32 35 AM" src="https://github.com/user-attachments/assets/7340f2c7-7b9d-48c6-9ed7-77b3ffa7846b" />

# Politician Face Classification Using Deep Learning

This project trains a **deep learning model** to classify **political leaders** using **transfer learning** on a **pretrained ResNet-18 model**. The dataset consists of cropped face images of well-known politicians, which are **automatically detected and extracted** using **MTCNN**.

---

##  **Project Workflow**
### 1 **Dataset Preparation**
- **Web Scraping with Scrapy**: Used **Scrapy** to collect images of political leaders from the GettyImage.
- **Structured dataset format**: Each subfolder contains images of a specific person.
- **Face Detection**: Uses **MTCNN** to crop faces before training.
- **Automatic Labeling**: Creates a `{name: ID}` dictionary for classification.

### 2 **Model Training**
- **Pretrained ResNet-18** → Fine-tuned for classification.
- **CrossEntropyLoss + Adam Optimizer** → Standard for multi-class classification.
- **Best Model Saving** → Tracks validation loss to save the optimal model.

### 3 **Inference on New Images**
- Loads **trained model** and applies transformations.
- Predicts the **political leader in a test image**.
- Displays **confidence score** of prediction.
