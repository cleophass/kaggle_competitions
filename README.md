# Machine Learning Projects

This repository contains my machine learning projects and Kaggle competition notebooks. Each project demonstrates practical applications of ML techniques on real-world datasets.  

---

## Projects

### 1. Titanic: Machine Learning from Disaster
- **Competition:** [titanic-machine-learning-from-disaster](https://www.kaggle.com/competitions/titanic/discussion/5105)
- **Description:** A classic Kaggle challenge aimed at predicting passenger survival on the Titanic. The project explores data analysis, feature engineering, and model selection to improve predictive performance.  
- **Techniques:**  
  - Data preprocessing and feature engineering  
  - Model training using LightGBM, XGBoost, and Random Forest
  - Evaluation using confusion matrix, recall, precision, and F1 score  

### 2. Predicting Road Accident Risk
- **Competition:** [Predicting Road Accident Risk](https://www.kaggle.com/competitions/playground-series-s5e10)
- **Description:** This project focuses on predicting the risk of road accidents using historical accident data. The goal is to build a model that identifies high-risk areas and times to improve road safety.  
- **Techniques:**  
  - Exploratory Data Analysis (EDA) and data cleaning  
  - Handling missing values and encoding categorical variables  
  - Model training using Decision Trees and Random Forest  
  - Evaluation using metrics such as RMSE and R²  
---

## Requirements

Python 3.x and the following libraries:

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score, f1_score, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
