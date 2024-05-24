# Project Overview

## Introduction
For this project, we will analyse a credit facility dataset, which can be accessed from https://www.kaggle.com/datasets/laotse/credit-risk-dataset?resource=download. Employing Python coding within Jupyter Notebook, we will construct a machine learning model called Artificial Neural Network (ANN). The main goal of this analysis is to predict the 'loan_status' column assessing the likelihood of a current customer defaulting on their debt payment. 

## Dataset Information

### Data Variables
The dataset consists of different details about customers' financial situations which include their personal information and factors related to loans. These details cover things like demographics, financial status and signs of how reliable they are with loans known as loan grade. These aspects form the basis of our analysis and the process of predicting outcomes (see dataset [here](https://github.com/Md-Khid/ANN_Classification_Prediction/blob/main/credit_risk_dataset.csv)).

### Data Pre-processing:

#### Loading and Categorising Loan Status in a Credit Risk Dataset

![1](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/74e0c63b-fae4-4830-90a2-a2bc5deea269)

#### Check Data Types 

![2](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/5e5fba1f-df18-45e6-bdb7-b3b8ff54a158)

#### Create Descriptive Stats table

![3](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/84593df1-73e6-4694-a293-1657cc54e8bc)

By creating a descriptive statistics table, we can summarise important details about the dataset including central measures and spread for both categories and numbers. From the table, we notice that most customers are renting and have a good credit rating of A for their loans which are mainly for education. With an average customer age of 27 years, it seems likely that many in the dataset are working adults pursuing further studies. Moreover, we observe that the predictor column (loan_status) has imbalanced data where roughly 78% of entries fall under the Non-Default category. Interestingly, there is an anomalous Max value of 123 years contained in the "person_emp_length" column which will be addressed by removing it as an outlier.


#### Identify Columns with Missing Values

![4](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/75816545-5fbd-4f55-9dcf-208cd45574a4)

Based on the output, it seems that the columns "person_emp_length" and "loan_int_rate" contain some missing values. To address this issue, we can decide on the most appropriate method for replacing the missing values. Possible approaches include using the mean, median or mode depending on the data distribution. Additionally, we will also examine for any special characters that might impede the machine learning algorithm process.

## Exploratory Data Analysis 
In this section, we will delve into comprehending the dataset. This encompasses tasks such as examining data distributions, identifying outliers, visualising correlations between variables and detecting any irregularities or trends, then transforming the insights obtained into valuable information.

#### Scaling Numerical Features

![6](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/d078cfbc-b1e3-4799-8d6c-52a80eef9c9e)

We will scale the numerical value columns in the dataset. This ensures that the variables are standardised to a consistent scale, thereby facilitating accurate interpretation of their relationships.

#### Correlation Matrix Plot

![7](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/daa6459c-5d18-41f3-b478-44a261598ea7)

Upon examining the correlation plot, it is apparent that there is not significant correlation among the column variables except for the 'cb_person_cred_hist_length' and 'person_age' columns.

#### Scatterplot

![8](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/7d5eec17-ed2f-4efb-9acc-275733890a94)

However, by colouring the plots with the categorical data columns, we can observe some interesting insights about the customers in the dataset. From the plots, we can observe the following points:

Based on the Loan Status plot: The plot suggests a higher proportion of individuals classified as non-defaulters compared to defaulters. Moreover, there is an observed trend suggesting that default rates tend to rise as loan amounts increase. Notably, loans characterised by smaller amounts and lower loan percent income exhibit a higher likelihood of being repaid resulting in a non-default status.

Based on the Home Ownership plot: The plot suggests that individuals who rent or hold mortgages often apply for larger loan amounts. Interestingly, most of the borrowers are from the mortgagors and renters group.

Based on the Loan Intent plot: It seems there is no apparent correlation between the purpose of the loan and either the loan amount or loan percent income. However, a notable concentration of data points is observed on the left-hand side of the plot. This concentration may suggest that the majority of borrowers, regardless of their loan intent, tend to seek loans with moderate amounts and are offered moderate loan amounts relative to their income.

Based on the Loan Grade plot: The plot shows that loans assigned higher grades (A and B) correspond to lower loan percent income, while loans with lower grades (E and F) entail higher loan percent income. This could suggest that higher grades (like A and B) tend to have borrowers who owe less relative to their income. On the other hand, loans with lower grades (E and F) have borrowers who owe more relative to their income, indicating higher financial risk.

Based on the Default on File plot: The plot suggests that individuals without a history of default tend to obtain loans across a broader range of amounts and generally at lower interest rates compared to those with a default record.


#### Bar Chart

![9](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/b964f51c-d149-4c0c-b21c-41d227419e6d)

Based on the Home Ownership chart: The data indicates a higher number of renters compared to homeowners or mortgage holders. This trend may stem from various factors. For example, younger individuals or those in transitional life stages may prefer renting over owning a home. Economic aspects like housing affordability and credit accessibility could also influence this choice.

Based on the Loan Intent chart: The majority of loans are designated for 'Education'. This trend may imply that many individuals, particularly younger ones pursuing further education, seek financial aid to manage the escalating costs of education.

Based on the Loan Grade chart: Most loans fall into grades 'A' and 'B'. This suggests that a considerable number of individuals maintain commendable creditworthiness. This likely reflects their financial prudence and effective debt management skills.

Based on the Loan Status chart: Analysis of 'Default' and 'Non-Default' statuses reveals that the majority of loans are 'Non-Default'. This indicates successful loan repayment management by most individuals. However, the presence of 'Default' loans underscores the financial challenges faced by some individuals.

Based on the Credit Bureau Default on File chart: A significant proportion of individuals have no defaults recorded with the credit bureau. This serves as another positive indicator of their creditworthiness and may suggest a history of responsible borrowing practices

#### Dummy Variables

![10](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/5c9e32f6-5986-4f49-ad63-812dd251cdeb)

By converting categorical variables into dummy variables and assigning the loan_status as (Non-Default=0, Default=1), we can prepare them for input into machine learning models. This ensures that categorical data seamlessly integrates into the modeling process, enhancing analysis and prediction tasks performed by the algorithms.

#### Remove Outliers

We will be excluding any outliers found in the dataset. This is because outliers can greatly affect the precision and reliability of the  ANN model.

#### Class Imbalance Correction Using SMOTE

![12](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/58f66107-ed1b-42b7-ae00-048d89b348d7)

To address the imbalance in the predictor column (loan_status) as mentioned earlier, we will use SMOTE to oversample the minority class (Default=1) to align with the majority class (Non-Default=0). Upon applying SMOTE, we can see that both classes now have equal balanced representation.

### ANN Model Training for Loan Status Prediction

For the modelling phase, we will split the dataset into training and testing datasets. We will use 70% of the data to train and 30% to test. To ensure reproducible results, we will set the random state to 42. We will intend to create an Artificial Neural Network (ANN) model comprising two concealed layers. 

#### Evaluation of Model - Confusion Matrix

![13](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/48dd6ece-3c98-4200-b778-a2085bae0e40)

To assess how well the ANN model performs, we employ various matrices to gauge its predictive ability. We will create a confusion matrix and analyse its performance metrics in predicting the loan_status outcome. Based on the matrix:

- Accuracy: Shows that 82% of all cases were accurately predicted by the model.
- Precision: Indicates that 87% of the predicted Default cases were accurately identified as true positives.
- Recall (Sensitivity): Shows that 76% of the actual Default cases were identified by the model.
- F1 Score: The model achieved a reasonable balance between precision and recall at 0.81. This suggests that it can effectively identify relevant instances (high recall) while also minimising false positives (high precision).
- Specificity: Indicates that 89% of Non-Default cases were correctly predicted by the model.

#### Evaluation of Model - Receiver Operating Characteristic (ROC) Curve and Area Under the Curve (AUC)

![14](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/4a226ba9-7ca0-42a8-ab92-0e34d7da1653)

We can delve deeper into assessing and demonstrating the effectiveness of the ANN model by creating a ROC and AUC chart. Based on the chart:

-ROC Curve: Both the training and testing ROC curves are notably above the diagonal red dashed line which signifies a no-skill classifier. This indicates that the model demonstrates strong predictive performance.

-AUC Value: The Area Under the Curve (AUC) for both the training and testing data is 0.90. This indicates that the model demonstrates a high degree of separability (where 0.50 denotes random chance) and can effectively differentiate between positive and negative classes.

## Evaluate Model Performance

#### Feed New Data Into Trained Model
![15](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/d6518c2e-e977-456b-b5dd-6bf8c3b347ea)

Using the [test](https://github.com/Md-Khid/ANN_Classification_Prediction/blob/main/Test.Data.csv) dataset, the trained ANN model accurately predicted about 85% for instances likely to be defaulters and non-defaulters. This indicates that the trained model generalised well to new data.

Predictor of Importance

![16](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/95e4dffa-17bf-44e8-9f3a-049622d6cc17)

As the ANN model is often seen as a 'black box,' it is important to determine the predictors that have the most impact on the model's predictions in order to gain insights into which inputs influence the model's decision-making process. This is particularly helpful for interpreting and identifying the key predictors that influence the model's predictions. Based to the features:

1.	person_home_ownership_RENT: This emerges as the most critical predictor. It suggests that the home ownership status particularly renting, has a significant influence on the model’s predictions. This could be due to various factors such as financial instability associated with renting. Thereby, increasing the risk of the event the model is predicting.
2.	loan_intent_PERSONAL: This feature ranks as the second most important. It indicates that the intent behind the loan, especially when it’s for personal use, strongly correlates with the outcome predicted by the model. This might be attributed to the fact that personal loans can be used for a variety of purposes, some of which might carry a higher risk.
3.	cb_person_cred_hist_length: This stands as the third most influential predictor. The length of a person’s credit history serves as a reliable indicator of their financial behaviour and reliability. Thereby, impacting the model’s predictions.
4.	person_age: This predictor is the fourth most significant. It suggests that the age of the person plays a substantial role in the model’s predictions. This could be due to various factors such as older individuals having more stable financial situations or younger individuals being more likely to take risks.
5.	loan_intent_EDUCATION: This feature ranking fifth in importance indicates that the intent behind the loan, particularly when it’s for educational purposes, significantly influences the model’s predictions. This may be attributed to factors such as the high cost of education which could increase the likelihood of the event the model is predicting.


### Concluding Remarks
Artificial Neural Networks (ANNs) offer a promising avenue for transforming credit risk assessment within the banking sector. However, it's crucial to recognise the vital role of human insight when analysing data. Even with advanced algorithms like ANNs, certain aspects of risk analysis rely on human judgment and intuition. Take for instance, the essential task of handling missing values and detecting outliers during data preprocessing. This critical step requires careful consideration and input from banking experts. Mishandling of missing values and outliers could distort results and potentially obscure valuable insights.

Additionally, improving the model's effectiveness involves integrating economic indicators into the datasets. This task demands a deep comprehension of financial markets, a knowledge that seasoned experts possess. While ANNs excel at processing large volumes of data, they often lack the contextual understanding needed to interpret variables accurately within the dynamic landscape of economic indicators. Therefore, the input and insights of human analysts are crucial in ensuring the meaningful interpretation and prediction of the model.


