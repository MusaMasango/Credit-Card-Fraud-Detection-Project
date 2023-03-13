## Credit-Card-Fraud-Detection-Project

## Loan prediction machine learning project

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
After downloading the data, I needed to clean it up so that it was usable for our model. I made the following changes
* Removed the Loan_ID columns from both datasets as it is not needed
