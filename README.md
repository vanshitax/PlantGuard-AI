# PlantGuard-AI

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![Accuracy](https://img.shields.io/badge/Accuracy-98.82%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

PlantGuard-AI is a deep learning-based plant disease detection system that identifies plant diseases from leaf images using a fine-tuned EfficientNetB0 model.

The application provides disease predictions, confidence scores, top-3 predictions, and disease information through an interactive Streamlit web interface.

## Features

* Classification of 38 plant disease categories
* Transfer learning using EfficientNetB0
* Fine-tuned model for improved accuracy
* Streamlit-based user interface
* Top-3 prediction display with confidence scores
* Disease information panel
* Image upload and real-time prediction

## Tech Stack

### Machine Learning
- TensorFlow
- Keras
- EfficientNetB0

### Application Development
- Streamlit

### Data Processing
- NumPy
- Pandas

### Visualization
- Matplotlib
- Seaborn

### Evaluation
- scikit-learn

## Dataset

Dataset: PlantVillage - New Plant Diseases Dataset

### Dataset Statistics

| Category          | Count  |
| ----------------- | ------ |
| Training Images   | 70,295 |
| Validation Images | 17,572 |
| Classes           | 38     |

The dataset contains healthy and diseased leaf images from multiple crops including tomato, potato, corn, grape, apple, strawberry, peach, pepper, soybean, orange, raspberry, and cherry.

## Model Architecture

Base Model: **EfficientNetB0**

Training Pipeline:

1. Transfer Learning using ImageNet pretrained weights
2. Feature extraction with frozen base layers
3. Fine-tuning upper EfficientNet layers
4. Model checkpointing and validation monitoring

Classification Head:

* GlobalAveragePooling2D
* Dropout
* Dense (38 classes)

## Results

| Metric              | Value  |
| ------------------- | ------ |
| Validation Accuracy | 98.82% |
| Validation Loss     | 0.0384 |

## Project Highlights

- Built a deep learning model capable of classifying 38 plant disease categories.
- Achieved 98.82% validation accuracy using transfer learning and fine-tuning.
- Utilized EfficientNetB0 pretrained on ImageNet for feature extraction.
- Developed an interactive Streamlit web application for real-time predictions.
- Displayed Top-3 predictions with confidence scores and disease information.
- Implemented data augmentation, early stopping, and model checkpointing.

## Application Preview

### Home Screen

![Home Screen](screenshots/app_home.png)

### Healthy Leaf Prediction

![Healthy Prediction 1](screenshots/healthy_prediction1.png)

![Healthy Prediction 2](screenshots/healthy_prediction2.png)

### Diseased Leaf Prediction

![Diseased Prediction 1](screenshots/diseased_prediction1.png)

![Diseased Prediction 2](screenshots/diseased_prediction2.png)

## Training Visualizations

### Training Curves

![Training Curves](screenshots/training_curves.png)

### Confusion Matrix

![Confusion Matrix](screenshots/confusion_matrix.png)

### Classification Report

![Classification Report](screenshots/classification_report.png)

## Project Structure

```text
PlantGuard-AI
│
├── notebooks/
├── screenshots/
├── test_images/
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/vanshitax/PlantGuard-AI.git

cd PlantGuard-AI

pip install -r requirements.txt
```

## Run Application

```bash
streamlit run streamlit_app.py
```

## Future Improvements

* Grad-CAM visual explanations
* Additional disease information coverage
* Mobile-friendly deployment
* Real-time camera integration

## Author

Vanshita Singh

B.Tech Information Technology
Manipal University Jaipur
