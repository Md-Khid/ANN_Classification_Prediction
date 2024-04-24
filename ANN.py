#!/usr/bin/env python
# coding: utf-8

# # Python Libraries and Modules for Data Preprocessing, Modelling, and Evaluation

# In[1]:


# Install pip packages if required
#!pip install imblearn
#!pip install tensorflow

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# Set a standard seaborn colour palette
sns.set_palette("colorblind")


# # Loading and Categorising Loan Status in a Credit Risk Dataset

# In[2]:


# Load and read the CSV file
df = pd.read_csv('credit_risk_dataset.csv')

# Map loan_status to categorical values
loan_status_mapping = {0: 'Non-Default', 1: 'Default'}
df['loan_status'] = df['loan_status'].map(loan_status_mapping)

# Display the updated DataFrame
df


# # Check data types 

# In[3]:


# Check data types of each column
df.dtypes


# # Section 4. Create Descriptive Stats table

# In[4]:


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


# # Identify Columns with Missing Values

# In[5]:


# Calculate number of missing values 
missing_values = df.isnull().sum()

# Filter the missing_values
columns_with_missing_values = missing_values[missing_values > 0]

# Display columns with missing values
columns_with_missing_values


# # Fill missing values

# In[6]:


# Fill missing values in 'person_emp_length' and 'loan_int_rate' columns with median
df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)
df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)

# Display number of columns with missing values
count_missing_values = df.isnull().sum().sum()
count_missing_values


# # Define special characters

# In[7]:


# Define special characters
special_chars = "!@#$%^&"

# Iterate over each column 
for column in df.columns:
    # Iterate over each row in current column
    for index, value in df[column].items():
        # Check if value contains any special characters
        if any(char in special_chars for char in str(value)):
            print(f"Special characters found in column '{column}', row {index}: {value}")


# # Scaling Numerical Features

# In[8]:


# Define the list of numeric column names
numeric_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']

# Apply Min-Max scaling to numerical columns
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
df


# # Section 9. Correlation Matrix Plot

# In[9]:


def plot_corr_and_print_highly_correlated(df):
    # Calculate correlation matrix for numeric columns
    corr_matrix = df.select_dtypes(include='number').corr()

    # Plot heatmap
    plt.figure(figsize=(10, 5))  # Specify the figure size
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", linewidths=.5, xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns, cbar_kws={'orientation': 'vertical'})
    plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
    plt.show()

    # Print highly correlated pairs
    print("Highly Correlated Features:")
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.8: 
                pair = (corr_matrix.columns[i], corr_matrix.columns[j])
                print(pair)

# Call the function
plot_corr_and_print_highly_correlated(df)


# # Scatterplot

# In[10]:


# Define the list of variables for hue
hue_variables = ['loan_status','person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(35, 15))

# Flatten axes for easier iteration
axes = axes.flatten()

# Iterate over hue variables and create subplots
for i, hue_var in enumerate(hue_variables):
    sns.scatterplot(data=df, x='loan_amnt', y='loan_int_rate', hue=hue_var, ax=axes[i])
    axes[i].set_xlabel('Loan Amount')
    axes[i].set_ylabel('Loan Percent Income')
    
    # Get handles and labels for the legend
    handles, labels = axes[i].get_legend_handles_labels()
    
    # Create a custom legend with a title
    axes[i].legend(handles=handles, labels=labels, title=hue_var, loc='center left', bbox_to_anchor=(1, 0.5))

# Hide empty subplots
for i in range(len(hue_variables), axes.size):
    fig.delaxes(axes.flatten()[i])

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# # Bar Charts

# In[11]:


# Select only the categorical columns
categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Define subplot layout
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12)) 

# Plot count for each unique item within each categorical column
for i, ax in enumerate(axes.flatten()):
    if i < len(categorical_columns):
        column = categorical_columns[i]
        order = df[column].value_counts().index  # Arrange the bars from highest to lowest count
        sns.countplot(x=column, data=df, order=order, ax=ax)
        ax.set_title('')
        ax.set_ylabel('')  # Remove the ylabel
        ax.tick_params(axis='x', rotation=45)  # Rotate x-labels by 45 degrees
    else:
        fig.delaxes(ax)  # Hide empty subplot

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()


# # Dummy Variables

# In[12]:


# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Remove loan_status from categorical_columns
categorical_columns.remove('loan_status')

# Perform one-hot encoding for categorical variables
df = pd.get_dummies(df, columns=categorical_columns)

# Map loan_status to {Non-Default=0, Default=1}
df['loan_status'] = df['loan_status'].map({'Non-Default': 0, 'Default': 1})

# Display the resulting DataFrame
df


# # Remove Outliers

# In[13]:


# Select numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Detect outliers using z-score
z_scores = stats.zscore(df[numeric_cols])
abs_z_scores = abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)

# Remove outliers from DataFrame
df = df[filtered_entries]

# Check the shape of the DataFrame after removing outliers
df.shape


# # Class Imbalance Correction Using SMOTE

# In[14]:


X = df.drop(columns=['loan_status'])  
y = df['loan_status']  

# Initialize SMOTE
smote = SMOTE()

# Perform SMOTE
X_resampled, y_resampled = smote.fit_resample(X, y)

# Save X_resampled and y_resampled to CSV
X_resampled.to_csv('X_resampled.csv', index=False)
y_resampled.to_csv('y_resampled.csv', index=False)

# Before SMOTE
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Before SMOTE')
counts = df['loan_status'].value_counts().sort_index()
counts.plot(kind='pie', autopct=lambda p: '{:.1f}% ({:,.0f})'.format(p, p * sum(counts) / 100))
plt.ylabel('')

# After SMOTE
plt.subplot(1, 2, 2)
plt.title('After SMOTE')
resampled_counts = y_resampled.value_counts().sort_index()
resampled_counts.plot(kind='pie', autopct=lambda p: '{:.1f}% ({:,.0f})'.format(p, p * sum(resampled_counts) / 100))
plt.ylabel('')

plt.show()


# # ANN Model Training for Loan Status Prediction

# In[15]:


# Separate features and target variable
X = X_resampled  
y = y_resampled 
    
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the ANN model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)


# # Evaluation of Model - Confusion Matrix

# In[16]:


# Predictions on test data
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)  # Convert probabilities to binary predictions

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

# Plot confusion matrix 
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16})
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

# Add legend box for metrics
metrics_legend = f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall (Sensitivity): {recall:.2f}\nF1 Score: {f1:.2f}\nSpecificity: {specificity:.2f}"
plt.text(1.40, 0.5, metrics_legend, fontsize=12, ha='left', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))

# Add total instances for each class below the legend box
total_0_instances = np.sum(conf_matrix[0])
total_1_instances = np.sum(conf_matrix[1])
plt.text(1.30, 0.3, f'Total Instances of 0: {total_0_instances}', fontsize=12, ha='left', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
plt.text(1.30, 0.2, f'Total Instances of 1: {total_1_instances}', fontsize=12, ha='left', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
plt.show()


# # Evaluation of Model - Receiver Operating Characteristic (ROC) Curve and Area Under the Curve (AUC)

# In[17]:


# Calculate ROC curve and AUC for training set
fpr_train, tpr_train, _ = roc_curve(y_train, model.predict(X_train))
auc_train = auc(fpr_train, tpr_train)

# Calculate ROC curve and AUC for testing set
fpr_test, tpr_test, _ = roc_curve(y_test, model.predict(X_test))
auc_test = auc(fpr_test, tpr_test)

# Plot both ROC curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, color='blue', lw=2, label='Train ROC curve (AUC = {:.2f})'.format(auc_train))
plt.plot(fpr_test, tpr_test, color='orange', lw=2, label='Test ROC curve (AUC = {:.2f})'.format(auc_test))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# # Save Trained Model

# In[18]:


# Save the trained model
model.save('ANN.model.h5')


# # Feed New Data Into Trained Model

# In[19]:


# Load the new data from CSV
test = pd.read_csv('Test.data.csv')

# Define the list of numeric column names
numeric_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']

# Apply Min-Max scaling to numerical columns
scaler = MinMaxScaler()
test[numeric_cols] = scaler.fit_transform(test[numeric_cols])

# Identify categorical columns
categorical_columns = test.select_dtypes(include=['object']).columns.tolist()

# Perform one-hot encoding for categorical variables
test = pd.get_dummies(test, columns=categorical_columns)

# Load the model and perform prediction
model = load_model('ANN.model.h5')

# Use X_test features for prediction
X_test = test.drop('loan_status', axis=1) 
predictions = model.predict(X_test)

# Convert probabilities into class labels
threshold = 0.5
class_predictions = [0 if pred < threshold else 1 for pred in predictions]

# Compare actual and predicted loan_status values
actual_loan_status = test['loan_status'].values
predicted_loan_status = class_predictions

# Create a DataFrame to compare actual and predicted loan_status values
comparison_df = pd.DataFrame({'Actual_loan_status': actual_loan_status, 'Predicted_loan_status': predicted_loan_status})

# Print the comparison DataFrame
comparison_df


# # Predictor of Importance

# In[20]:


# Load the saved model
model = keras.models.load_model('ANN.model.h5')

# Get the column names from the DataFrame
column_names = X_resampled.columns

# Retrieve the weights of the first layer of the neural network model
weights = model.layers[0].get_weights()[0]

# Calculate the absolute sum of weights for each feature
feature_weights_sum = abs(weights).sum(axis=0)

# Create a dictionary of feature names and their importance
feature_importance = dict(zip(column_names, feature_weights_sum))

# Sort the dictionary by value in descending order
sorted_feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))

# Select the top 5 features
top_5_features = dict(list(sorted_feature_importance.items())[:5])

# Create a horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(list(top_5_features.keys()), list(top_5_features.values()))
plt.xlabel('Importance')
plt.ylabel('Predictors')
plt.title('Top 5 Features Importance')
plt.gca().invert_yaxis() 
plt.show()


# In[ ]:




