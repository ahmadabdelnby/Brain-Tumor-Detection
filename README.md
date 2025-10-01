# Brain Tumor Classification Machine Learning Project

## Overview
This Jupyter notebook (`notebook2f4cccb559.ipynb`) implements a comprehensive machine learning pipeline for brain tumor classification using multiple algorithms. The project utilizes texture analysis features extracted from medical images to predict the presence or absence of brain tumors.

## Dataset
- **File**: `Brain Tumor.csv`
- **Size**: 3,764 samples
- **Features**: 12 texture-based features extracted from medical images
- **Target Variable**: `Class` (Binary classification: 0 = No Brain Tumor, 1 = Brain Tumor)

### Feature Description
The dataset includes the following texture analysis features:
- **Mean**: Average intensity value
- **Variance**: Measure of intensity variation
- **Standard Deviation**: Square root of variance
- **Entropy**: Measure of randomness/disorder
- **Skewness**: Measure of asymmetry in distribution
- **Kurtosis**: Measure of tail heaviness in distribution
- **Contrast**: Measure of local intensity variation
- **Energy**: Measure of textural uniformity
- **ASM (Angular Second Moment)**: Square of energy
- **Homogeneity**: Measure of local uniformity
- **Dissimilarity**: Measure of local variation
- **Correlation**: Measure of linear dependency between pixels

## Machine Learning Pipeline

### 1. Data Preprocessing
- **Data Loading**: Import CSV file using pandas
- **Exploratory Data Analysis**: 
  - Dataset shape analysis
  - Statistical summary
  - Null value detection
  - Class distribution analysis
- **Feature Scaling**: StandardScaler normalization to ensure all features contribute equally

### 2. Data Splitting
- **Train-Test Split**: 80% training, 20% testing
- **Stratified Sampling**: Maintains class distribution in both sets
- **Random State**: Set to 2 for reproducible results

### 3. Machine Learning Models

#### Support Vector Machine (SVM)
- **Kernel**: Linear kernel for binary classification
- **Training**: Fit on standardized training data
- **Evaluation**: Accuracy score, confusion matrix, and classification report

#### Logistic Regression
- **Algorithm**: Standard logistic regression
- **Training**: Fit on standardized training data
- **Evaluation**: Accuracy score on both training and test sets

#### K-Nearest Neighbors (KNN)
- **Parameters**: 
  - n_neighbors = 5
  - algorithm = 'auto'
  - leaf_size = 50
- **Evaluation**: Accuracy score on test set

### 4. Ensemble Prediction System
The notebook implements a voting-based ensemble approach:
- Combines predictions from all three models (SVM, Logistic Regression, KNN)
- Final prediction based on majority vote (≥2 models predicting positive)
- Provides more robust and reliable predictions

## Key Features

### Model Evaluation
- **Training Accuracy**: Measures model performance on training data
- **Test Accuracy**: Measures model generalization on unseen data
- **Confusion Matrix**: Detailed breakdown of correct/incorrect predictions
- **Classification Report**: Precision, recall, and F1-scores

### Prediction System
- **Real-time Prediction**: Capability to classify new patient data
- **Data Standardization**: Ensures new data follows same preprocessing pipeline
- **Ensemble Decision**: Uses multiple models for final classification

## Usage

### Prerequisites
```python
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

### Running the Notebook
1. Ensure `Brain Tumor.csv` is in the same directory
2. Execute cells sequentially for complete pipeline
3. Use the prediction system with new patient data

### Making Predictions
To classify a new patient, provide texture features in the following format:
```python
# Example patient data
data = (62, 0, 138, 294, 1, 1, 106, 0, 1.9, 1, 3, 2)
```

## Results
The notebook evaluates three different machine learning algorithms and combines their predictions for improved accuracy. The ensemble approach helps reduce false positives/negatives in brain tumor detection, which is critical for medical diagnosis applications.

## File Structure
```
Machine Learning/
├── notebook2f4cccb559.ipynb    # Main notebook file
├── Brain Tumor.csv             # Dataset
└── README_notebook2f4cccb559.md # This documentation
```

## Notes
- All models are trained on standardized features for optimal performance
- The ensemble approach provides additional confidence in predictions
- Results include detailed evaluation metrics for medical assessment
- The notebook is designed for educational and research purposes

## Future Enhancements
- Cross-validation for more robust model evaluation
- Feature importance analysis
- ROC curves and AUC metrics
- Hyperparameter tuning for optimal performance
- Additional ensemble methods (Random Forest, Gradient Boosting)

---

*This project demonstrates the application of machine learning techniques in medical image analysis and diagnosis assistance.*
