# Project Overview

## Introduction
For this project, we will analyse a credit facility dataset, which can be accessed from https://www.kaggle.com/datasets/laotse/credit-risk-dataset?resource=download. Employing Python coding within Jupyter Notebook, we will construct a machine learning model called Artificial Neural Network (ANN). The main goal of this analysis is to predict the 'loan_status' column assessing the likelihood of a current customer defaulting on their debt payment. 

## Dataset Information

### Data Variables
The dataset consists of different details about customers' financial situations which include their personal information and factors related to loans. These details cover things like demographics, financial status and signs of how reliable they are with loans known as loan grade. These aspects form the basis of our analysis and the process of predicting outcomes . [Dataset](https://github.com/Md-Khid/ANN_Classification_Prediction/blob/main/credit_risk_dataset.csv)


### Data Dictionary
| Variable                    | Description                                     |
|-----------------------------|-------------------------------------------------|
| person_age                  | Age                                             |
| person_income               | Annual Income                                   |
| person_home_ownership       | Home Ownership: (Mortgage, Other Own, Rent)     |
| person_emp_length           | Employment Length (in years)                    |
| loan_intent                 | Loan Intent: (Debt Consolidation, Education, Home Improvement, Medical, Personal, Venture) |
| loan_grade                  | Loan Grade: (A, B, C, D, E, F, G)               |
| loan_amnt                   | Loan Amount                                     |
| loan_int_rate               | Interest Rate                                   |
| loan_status                 | Loan Status: (0 = Non Default, 1 = Default)     |
| loan_percent_income        | Percent Income                                  |
| cb_person_default_on_file  | Credit Bureau-History Default: (No, Yes)        |
| cb_person_cred_hist_length | Credit Bureau-Credit History Length             |


## Data Preparation

In this phase of data processing, we will refine the dataset for analysis by addressing missing values, handling special characters, and encoding variables. Additionally, we will import all necessary modules and libraries for the project and transform categorical variables into category columns for data visualisation purposes.

### Data Pre-processing:

#### Import Python Libraries and Modules for Data Preprocessing, Modelling and Evaluation
```
# Install pip packages if required
#!pip install imblearn
#!pip install tensorflow

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
```

#### Loading and Categorising Loan Status in a Credit Risk Dataset

```
# Load and read the CSV file
df = pd.read_csv('credit_risk_dataset.csv')

# Map loan_status to categorical values
loan_status_mapping = {0: 'Non-Default', 1: 'Default'}
df['loan_status'] = df['loan_status'].map(loan_status_mapping)

# Display the updated DataFrame
df
```
![1](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/d7246f0f-7126-4bbe-891b-344078058cb9)


#### Check data types of each column
```
# Check data types of each column
df.dtypes
```

![2](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/0515c36a-a3fb-49c6-9337-e2667c17954e)

#### Create Descriptive Stats table
By creating a descriptive statistics table, we can summarise important details about the dataset including central measures and spread for both categories and numbers. From the table, we notice that most customers are renting and have a good credit rating of A for their loans which are mainly for education. With an average customer age of 27 years, it seems likely that many in the dataset are working adults pursuing further studies. Moreover, we observe that the predictor column (loan_status) has imbalanced data, where roughly 78% of entries fall under the Non-Default category. 
```
# Create Descriptive Stats table 
Descriptive_Stats = df.describe(include='all').round(2)

# Separate columns into categorical and numerical groups
categorical_columns = Descriptive_Stats.select_dtypes(include=['object']).columns
numeric_columns = Descriptive_Stats.select_dtypes(exclude=['object']).columns

# Order columns (categorical followed by numerical)
ordered_columns = list(categorical_columns) + list(numeric_columns)
Descriptive_Stats = Descriptive_Stats.reindex(ordered_columns, axis=1)

# Transpose Descriptive Stats table 
Descriptive_Stats = Descriptive_Stats.transpose()

Descriptive_Stats
```

![3](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/89902bff-0a51-4d02-93a6-47cbedd2e12d)


#### Identify Columns with Missing Values
```
# Calculate number of missing values 
missing_values = df.isnull().sum()

# Filter the missing_values
columns_with_missing_values = missing_values[missing_values > 0]

# Display columns with missing values
columns_with_missing_values
```

![4](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/06593c2d-5caa-4415-8515-8ed4dac12adc)



