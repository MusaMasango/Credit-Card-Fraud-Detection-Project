## Credit-Card-Fraud-Detection-Project

## Introduction
CLosses related to credit card fraud will grow to $43 billion within five years and climb to $408.5 billion globally within the next decade, according to a recent Nilson Report — meaning that credit card fraud detection has become more important than ever. 

The sting of these rising costs will be felt by all parties within the payment lifecycle: from banks and credit card companies who foot the bill of such fraud, to the consumers who pay higher fees or receive lower credit scores, to merchants and small businesses who are slapped with chargeback fees.

With digital crime and online fraud of all kinds on the rise, it’s more important than ever for organizations to take firm and clear steps to prevent payment card fraud through advanced technology and strong security measures.
 
Credit card fraud is the act of using another person’s credit card to make purchases or request cash advances without the cardholder’s knowledge or consent. These criminals may obtain the card itself through physical theft, though increasingly fraudsters are leveraging digital means to steal the credit card number and accompanying personal information to make illicit transactions.

There is some overlap between identity theft and credit card theft. In fact, credit card theft is one of the most common forms of identity theft. In such cases, a fraudster uses an individual’s personal information, which is often stolen as part of a cyberattack or data breach, to open a new account that the victim does not know about. This activity is considered both identity fraud and credit card fraud.

## Objective

In this machine learning project, we solve the problem of detecting credit card fraud transactions using machine numpy, scikit learn, and few other python libraries. We overcome the problem by creating a binary classifier and experimenting with various machine learning techniques to see which fits better.

## Stakeholders

The results obtained from this project can be used by various stakeholders within the bank such as
* Credit risk department
* Credit analysts
* Bank fraud team
* Cybersecurity team

## Importance of the project

For any bank or financial organization, credit card fraud detection is of utmost importance. We have to spot potential fraud so that consumers can not bill for goods that they haven’t purchased. The aim is, therefore, to create a classifier that indicates whether a requested transaction is a fraud.

## Code and Resources used

**Python Version**:3.9.12 

**Packages**:Pandas,Numpy,Scikit learn,Matplotlib,Seaborn,Imblearn, Collection, Intertools

**Data Source**:https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Data Collection
The datasets used in this project were downloaded from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud. I then read the two csv files using the pd.read_csv() command.

## Data Cleaning
After downloading the data, I needed to clean it up so that it was usable for our model. In our case, the dataset did not contain any missing values and the data was of the correct format.

## Exploratory Data Analysis (EDA)
The data only consists of numerical variables, no categorical variables were present. I looked at different distributions for the numeric data. Below are highlights from the data visualization section

![bar graph](https://github.com/MusaMasango/loan-prediction-machine-learning-project/blob/main/bar%20graph.png)


## Model Building 
First I transformed categorical variables into dummy variables. I also split the data into train and test data sets with a test size of 30%. 

I tried 3 different models and evaluated them using the accuracy score.

The 3 different models used are:
* Logistic regression Classifier 
* Decision tree Classifier
* Random forest Classifier

The reason why I chose this models is beacause since we are dealing with a classification problem these models work best with categorical variables. In addition, these models are easy to implement.

## Model Performance
The logistic regression model far outperformed the the other approaches on the test and validation sets
* Decision tree : Accuracy score = 66.67%
* Random forest : Accuracy score = 77.78%
* Logistic regression : Accuracy score = 78.47%

This results makes sense intuitively, since logistic regression algorithm works best where the target variable (dependant variable) is a binary, in this case since the loan status is a binary value between 0 and 1, the logistic regression algorithm will perform better compared to the other models.

## Conclusion
1. Credit_History is a very important variable  because of its high correlation with Loan_Status therefore showing high Dependancy for the latter.
2. The Logistic Regression algorithm is the most accurate: **approximately 78%**.
3. This project showed how traditional machine learning approaches such as Random forest and Logistic regression perform well on a standard dataset. Depending on the type of dataset, in reality, these models will surely give a competitive performance. In other cases, like the ones where regular payments over a while are a deciding factor, time-series models such as RNNs or LSTMs would perform better.  
This project shows the importance and relevance of using machine learning for loan prediction. We saw some existing approaches and datasets used to approach loan eligibility prediction and how AI might help smoothen this process. Finally, we built an end-to-end loan prediction machine learning project using a publicly available dataset from scratch. At the end of this project, one would know how different features influence the model prediction and how specific attributes affect the decision more than the other features.

