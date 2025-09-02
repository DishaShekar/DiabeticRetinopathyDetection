### 🩺 Diabetic Retinopathy Classification

This project focuses on classifying Diabetic Retinopathy (DR) from retinal images using deep learning techniques. The model is trained on a pre-processed dataset (Gaussian filtered fundus images) and can classify between different stages of DR.

## 📌 Dataset

The dataset used is from Kaggle:
🔗 Diabetic Retinopathy 224x224 Gaussian Filtered

It contains:

Retinal fundus images resized to 224x224 pixels

CSV file with image IDs and labels (0-4 for severity levels)

Class labels:

0 → No_DR

1 → Mild

2 → Moderate

3 → Severe

4 → Proliferative_DR

Additionally, a binary version (No_DR vs DR) is also created.

## ⚙️ Project Workflow

Data Preprocessing

Loaded image data and labels from CSV

Mapped severity levels to categorical labels (No_DR, Mild, etc.)

Visualized class distribution

Image Augmentation

Used ImageDataGenerator for real-time augmentation

Applied transformations (rotation, zoom, flips, etc.)

Model Development

Built using TensorFlow/Keras

CNN-based architecture with Conv2D, MaxPooling, Dropout, Dense layers

Optimized using Adam optimizer and categorical crossentropy

Evaluation

Accuracy, Loss curves visualization

Confusion matrix for class-wise performance

Export

Saved the trained model (.h5 / .tflite) for deployment

## 🛠️ Tech Stack

Python

TensorFlow / Keras

NumPy, Pandas, Matplotlib

Scikit-learn

OpenCV

## 📊 Results

The model successfully classifies diabetic retinopathy into multiple stages.

Accuracy and loss plots show effective training.

Class distribution and confusion matrix give insights into model performance.

## 🚀 How to Run

Clone this repository

git clone <your-repo-link>
cd diabetic-retinopathy-classification


Install dependencies

pip install -r requirements.txt


Download dataset from Kaggle
 and place it inside data/

Run the Jupyter Notebook / Python script

jupyter notebook


or

python train.py

## 📌 Future Work

Experiment with Transfer Learning (ResNet, EfficientNet, InceptionV3)

Improve class balance using SMOTE / advanced augmentation

Deploy as a web app using Streamlit / FastAPI
