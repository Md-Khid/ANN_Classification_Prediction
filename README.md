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


