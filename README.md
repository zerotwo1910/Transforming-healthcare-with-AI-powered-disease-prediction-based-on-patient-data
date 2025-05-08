# Transforming-Healthcare-with-AI-Powered-Disease-Prediction-Based-on-Patient-Data

![Heart Disease Prediction](https://img.shields.io/badge/Project-AI%20Healthcare-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)

## Overview
This project develops an AI-powered system for predicting heart disease using machine learning classification techniques. By analyzing patient clinical data, the system determines the presence or absence of heart disease, providing a valuable tool for early detection and diagnosis support.

## Problem Statement
Heart disease remains the leading cause of death worldwide, with many cases going undetected until serious complications arise. Early detection through AI-assisted diagnosis can significantly improve patient outcomes while reducing healthcare costs through preventive care. This project addresses a binary classification problem in medical diagnostics with significant real-world applications.

## Key Features
- **Multiple ML Models**: Implements and compares Logistic Regression, Random Forest, Neural Network, and Gradient Boosting
- **High Accuracy**: Achieves ≥85% prediction accuracy on test data
- **Optimized for Clinical Use**: Prioritizes recall to minimize missed disease cases
- **Interpretable Results**: Provides feature importance analysis and visualization
- **User-Friendly Interface**: Includes GUI for clinical staff to make predictions

## Dataset
The project uses the Heart Disease Dataset from Kaggle, containing 303 patient records with 13 clinical features:
- Demographics: age, sex
- Vital signs: resting blood pressure, cholesterol levels
- ECG results: resting ECG, maximum heart rate
- Exercise test results: exercise-induced angina, ST depression
- And more health indicators

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | 0.82 | 0.83 | 0.80 | 0.81 | 0.87 |
| Random Forest | 0.85 | 0.86 | 0.83 | 0.84 | 0.90 |
| Gradient Boosting | 0.87 | 0.88 | 0.85 | 0.86 | 0.92 |
| Neural Network | 0.86 | 0.87 | 0.84 | 0.85 | 0.91 |

## Project Workflow
```
1. Data Preprocessing
   - Handle missing values
   - Encode categorical variables
   - Remove outliers
   - Scale features
   
2. Exploratory Data Analysis
   - Univariate analysis
   - Bivariate analysis
   - Correlation studies
   
3. Feature Engineering
   - Feature creation & transformation
   - Feature selection
   
4. Model Development
   - Training multiple algorithms
   - Hyperparameter tuning
   - Cross-validation
   
5. Evaluation & Visualization
   - Performance metrics
   - ROC curves
   - Feature importance
   
6. Deployment
   - GUI development
   - Model serialization
```

## Key Insights
- ST depression during exercise (oldpeak) is the strongest predictor
- Chest pain type, thalassemia, and maximum heart rate are significant indicators
- Models achieve strong sensitivity for disease detection
- Feature importance aligns with established medical knowledge

## Technologies Used
- **Python 3.9**
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn, xgboost
- **Deep Learning**: tensorflow, keras
- **GUI Development**: tkinter
- **Model Serialization**: pickle

## Installation & Usage

```bash
# Clone this repository
git clone https://github.com/zerotwo1910/Transforming-healthcare-with-AI-powered-disease-prediction-based-on-patient-data.git

# Navigate to project directory
cd Main_Program
# Install dependencies
pip install -r requirements.txt

# Run the application
python AI-Powered_Healthcare_Disease_Prediction_System_(Final Draft).py
```

## Screenshots
| Description | Screenshot |
|-------------|------------|
| Main Interface | ![Main Interface](Screenshots/main_interface.png) |
| EDA Tab | ![EDA Tab](Screenshots/eda_tab.png) |
| Preprocess & Data Modeling | ![Preprocess & Data Modeling](Screenshots/preprocess_and_datamodeling.png) |
| ROC Visual | ![ROC Visual](Screenshots/roc_visualization.png) |
| Confusion Matrix | ![Confusion Matrix](Screenshots/confusion_matrix.png) |
| Correlation Matrix | ![Correlation Matrix](Screenshots/correlation_martix.png) |
| Feature Importance | ![Feature Importance](Screenshots/feature_importance.png) |
| Patient Distribution | ![Patient Distribution](Screenshots/patient_distribution.png) |
| Manual Prediction | ![Manual Prediction](Screenshots/manual_prediction.png) |

## Future Work
- Integration of additional clinical parameters
- Implementation of explainable AI techniques
- Development of web-based application
- Clinical validation studies
- Transfer learning with larger medical datasets
- Real-time prediction capabilities

## Contributors
- [Sriram Kumar K](https://github.com/zerotwo1910)
- [Team Member 2](https://github.com/teammember2) (if applicable)
- [Team Member 3](https://github.com/teammember3) (if applicable)

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgements
- [Kaggle](https://www.kaggle.com) for providing the dataset
- Sri Ramanujar Engineering College for project guidance
- All libraries and tools used in this project
