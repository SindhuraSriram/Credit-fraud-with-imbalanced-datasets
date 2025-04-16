# Credit Fraud Detection: Dealing with Imbalanced Datasets  

## Overview  
This Jupyter Notebook explores techniques for handling imbalanced datasets in credit fraud detection. Fraudulent transactions are often rare compared to legitimate ones, making it challenging for machine learning models to detect them effectively. This notebook demonstrates various approaches to address class imbalance and improve fraud detection performance.  

## Dataset  
The dataset used in this notebook contains anonymized transaction data, with features representing different transaction attributes. The target variable indicates whether a transaction is fraudulent (`1`) or legitimate (`0`).  

Dataset URL - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Key Topics Covered  
### 1. Exploratory Data Analysis (EDA)  
- Understanding the distribution of fraud vs. non-fraud transactions  
- Visualizing feature distributions  

### 2. Handling Class Imbalance  
- Oversampling using **SMOTE (Synthetic Minority Over-sampling Technique)**  
- Undersampling the majority class  
- Combining over- and under-sampling  

### 3. Model Training & Evaluation  
- Training machine learning models (e.g., **Logistic Regression, Random Forest, XGBoost**)  
- Evaluating model performance using **precision, recall, F1-score, and AUC-ROC**  
- Addressing bias due to class imbalance  

### 4. Hyperparameter Tuning  
- Using **GridSearchCV** or **RandomizedSearchCV** for optimal parameter selection  

### 5. Comparing Different Approaches  
- Baseline model vs. models trained with resampling techniques  
- Impact of data balancing on fraud detection accuracy  

## Requirements  
To run this notebook, install the following dependencies:  
```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn xgboost
