# Company Bankruptcy Prediction Using Machine Learning Models

**Author**: Chi-Lin Li  
**Date**: May 3, 2024

## Project Overview

This project focuses on predicting corporate bankruptcy using machine learning models applied to financial datasets. Given the imbalanced nature of the data, the **Synthetic Minority Over-sampling Technique (SMOTE)** is utilized to address class imbalance, followed by rigorous hyperparameter tuning to improve model performance. The project implements several machine learning models, including **XGBoost**, **LightGBM**, and **Multi-Layer Perceptron (MLP)**, evaluating them based on **accuracy**, **F1-score**, and the **area under the receiver operating characteristic (AUC) curve**.

### Key Features:
- **Class Imbalance Handling**: Utilization of SMOTE for oversampling the minority class to improve the model’s ability to predict rare bankruptcy events.
- **Model Evaluation**: Comparison of XGBoost, LightGBM, and MLP models with hyperparameter optimization through **GridSearchCV**.
- **Performance Metrics**: Focus on accuracy, F1-score, and AUC to evaluate and compare model performance.

## Methodologies

### 1. Data Preprocessing
- **Missing Value Handling**: Columns with missing values are identified, and appropriate strategies (e.g., imputation) are applied.
- **Normalization**: Financial metrics are normalized using **StandardScaler** to bring all numerical variables to the same scale, ensuring unbiased model training.
- **SMOTE**: The **Synthetic Minority Over-sampling Technique** generates synthetic samples from the minority class to address class imbalance, enhancing model performance for predicting rare bankruptcy events.

### 2. Machine Learning Models
- **XGBoost**: A decision tree-based ensemble learning algorithm using gradient boosting, optimized for loss reduction.
- **LightGBM**: A gradient boosting framework that grows trees leaf-wise, optimized for large datasets with high accuracy and efficiency.
- **Multi-Layer Perceptron (MLP)**: A neural network model with multiple layers of neurons, trained using backpropagation to predict complex patterns in the data.

### 3. Hyperparameter Optimization
- **GridSearchCV**: Applied to each model to find the best hyperparameters, optimizing for the **F1-score** to balance precision and recall.

## Results

### XGBoost
- **Accuracy**: 97.16%
- **F1-Score**: 0.5753 (after optimization)
- **AUC**: 0.9508

### LightGBM
- **Accuracy**: 97.80%
- **F1-Score**: 0.6250 (after optimization)
- **AUC**: 0.9628  
  *LightGBM performed the best overall, with the highest F1-score and AUC, demonstrating strong classification capabilities.*

### Multi-Layer Perceptron (MLP)
- **Accuracy**: 96.79%
- **F1-Score**: 0.4615 (after optimization)
- **AUC**: 0.9274

### Feature Importance
- **Top Features Identified by XGBoost and LightGBM**:
  - Net Value Per Share (C)
  - Quick Assets/Current Liability
  - Working Capital to Total Assets
  - Realized Sales Gross Margin
  - Regular Net Profit Growth Rate

## Conclusion

This project explored three machine learning models for bankruptcy prediction, with **LightGBM** emerging as the best-performing model after hyperparameter tuning. Its high F1-score and AUC demonstrate its superior ability to classify and predict company bankruptcy based on financial indicators. The results highlight the importance of proper data preprocessing, model selection, and optimization techniques in achieving accurate financial predictions.

## Installation

To replicate this project, install the following dependencies:

```bash
pip install numpy pandas scikit-learn xgboost lightgbm imbalanced-learn
```

## Usage

1. **Preprocess Data**: Load and preprocess the financial dataset, handling missing values, normalizing data, and applying SMOTE for class balancing.
2. **Train Models**: Train XGBoost, LightGBM, and MLP models on the training set.
3. **Optimize Models**: Use GridSearchCV for hyperparameter tuning and evaluate model performance based on F1-score and AUC.
4. **Analyze Results**: Compare the models using performance metrics and feature importance, and select the best-performing model for final predictions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contribution

Chi-Lin Li contributed 100% to this project.
