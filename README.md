# Political Leader Image Classification using Machine Learning

## Overview
This project implements a **machine learning-based image classification system** to recognize **political figures** using **facial recognition techniques**. The dataset consists of images scraped from **GettyImages** for world leaders such as **Joe Biden, Vladimir Putin, Donald Trump, Justin Trudeau, and Xi Jinping**. The project leverages **Scrapy** for data collection, **OpenCV** for image processing, and **Support Vector Machine (SVM)** for classification.

## Project Workflow
The project follows a structured pipeline consisting of five key components:

### 1. Data Collection (Web Scraping)
- Utilized **Scrapy**, a Python web scraping framework, to **extract and download images** of political leaders from GettyImages.
- The scraper was configured to collect a fixed number of images per leader while avoiding **robots.txt restrictions**.

### 2. Image Preprocessing (OpenCV)
- Applied **Haar Cascade Classifiers** to **detect and crop** faces in the images.
- Used **eye detection** to refine the **cropping process**, ensuring only valid facial regions were retained.
- Converted images to grayscale for consistent preprocessing.

### 3. Feature Extraction
- Performed **Wavelet Transform** to extract high-quality facial features.
- Resized images to a uniform size for **standardized feature representation**.

### 4. Model Training (Support Vector Machine - SVM)
- Extracted **principal features** from preprocessed images.
- Implemented **SVM** as the classifier, fine-tuned using **GridSearchCV** to optimize performance.
- Applied **K-Fold Cross Validation** to evaluate and improve model robustness.

### 5. Prediction & Evaluation
- Given a new image, the trained model predicts the **political leader**.
- Accuracy was assessed based on **classification performance metrics**.

## Technologies Used
- **Python** (Primary programming language)
- **Scrapy** (Web scraping)
- **OpenCV** (Face and eye detection)
- **NumPy & Pandas** (Data manipulation)
- **Matplotlib** (Visualization)
- **Scikit-learn** (Machine learning model training)
- **Jupyter Notebook** (Development environment)

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo-name/political-leader-classification.git
cd political-leader-classification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
