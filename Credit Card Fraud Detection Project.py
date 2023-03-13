#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for statistical data visualization
get_ipython().run_line_magic('matplotlib', 'inline')
from collections import Counter
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report ,confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score


# ### Import dataset

# In[2]:


# Import dataset
df = pd.read_csv("creditcard.csv")


# ### Exploratory data analysis

# In[3]:


# view dimensions of the dataset

df.shape


# We can see that there are 284807 instances and 31 attributes in the data set.

# In[4]:


# preview the dataset

df.head()


# In[5]:


# summary of the dataset

df.info()


# We can see that there are no missing values in the dataset and the data is of the correct type

# In[6]:


# statistical summary of the dataset

df.describe(include='all')


# ### Types of variables

# In[7]:


# Explore variables in the dataset

# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :\n\n', categorical)


# As we can see, the dataset does not have any categorical variables

# In[8]:


# find numerical variables

numerical = [var for var in df.columns if df[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)


# In[9]:


# view the numerical variables

df[numerical].head()


# Summary of numerical variables
# 
# (a) There are 6 numerical variables.
# 
# (b) All of the numerical variables are of float64 data type, with 30 of these variable being floats and only one variable being an integer.
# 

# In[10]:


# check missing values in numerical variables

df[numerical].isnull().sum()


# We can see that all the 31 numerical variables do not contain missing values.

# ### Declare feature vector and target variable

# In[11]:


# declare feature vector and target variable

X = df.drop(['Class'], axis=1)

y = df['Class']


# In[12]:


# let’s check the number of occurrences of each class label and plot the information using matplotlib.

non_fraud = len(df[df.Class == 0])
fraud = len(df[df.Class == 1])
fraud_percent = (fraud / (fraud + non_fraud)) * 100
print("Number of Genuine transactions: ", non_fraud)
print("Number of Fraud transactions: ", fraud)
print("Percentage of Fraud transactions: {:.4f}".format(fraud_percent))


# In[13]:


import matplotlib.pyplot as plt
labels = ["Genuine", "Fraud"]
count_classes = df.value_counts(df['Class'], sort= True)
count_classes.plot(kind = "bar", rot = 0)
plt.title("Visualization of Labels")
plt.ylabel("Count")
plt.xticks(range(2), labels)
plt.show()


# We can observe that the genuine transactions are over 99%! This is not good.

# In[14]:


# Let’s apply scaling techniques on the “Amount” feature to transform the range of values. We drop the original “Amount” column and add a new column with the scaled values. We also drop the “Time” column as it is irrelevant.

scaler = StandardScaler()
df["NormalizedAmount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
df.drop(["Amount", "Time"], inplace= True, axis= 1)


# ### Data Visualization 

# In[17]:


#List of all numeric columns
num = df.select_dtypes('number').columns.to_list()
num


# In[22]:


# numeric df
credit_num =  df[num]
credit_num


# In[23]:


for col in credit_num:
    plt.hist(credit_num[col])
    plt.title(col)
    plt.show()


# ### Correlation matrix

# In[20]:


#plotting the correlation matrix
sns.heatmap(df.corr() ,cmap='coolwarm')


# ### Correlation table for a more detailed analysis:

# In[21]:


#correlation table
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# It is evident from the above correlation matrix that the column V11 has the highest correlation with the Class(a positive correlation of 0.15). Therefore our target variable (Class) is highly dependent on this column

# ### Split data into separate training and test set 

# In[15]:


# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[16]:


# check the shape of X_train and X_test

X_train.shape, X_test.shape


# In[17]:


# check the shape of y_train and y_test

y_train.shape, y_test.shape


# ### Feature engineering

# In[18]:


# check data types in X_train

X_train.dtypes


# In[19]:


# display numerical variables

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical


# In[20]:


# check missing values in X_train

X_train.isnull().sum()


# In[21]:


# check missing values in X_test

X_test.isnull().sum()


# We can see that there are no missing values in X_train and X_test.

# ### Addressing the Class-Imbalance issue

# In[22]:


# Address the Class-Imbalance issue using the SMOTE technique
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X, y)
print("Resampled shape of X: ", X_resampled.shape)
print("Resampled shape of y: ", y_resampled.shape)
value_counts = Counter(y_resampled)
print(value_counts)
(X_train, X_test, y_train, y_test) = train_test_split(X_resampled, y_resampled, test_size= 0.3, random_state= 42)


# We can see that the data is now balanced

# ### Feature scaling

# In[23]:


# We use feature scaling to ensure that that the values in the data are scaled equally
cols = X_train.columns

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# In[24]:


X_train = pd.DataFrame(X_train, columns=[cols])


# In[25]:


X_test = pd.DataFrame(X_test, columns=[cols])


# In[26]:


X_train.head()


# We now have X_train dataset ready to be fed into the classifiers. This will be done in the following section

# ### Model training

# In[27]:


# train a logistic regression classifier on the training set
from sklearn.linear_model import LogisticRegression


# instantiate the model
LR = LogisticRegression(max_iter=100)


# fit the model
LR.fit(X_train, y_train)

# predict values using the test data
y_pred_1 = LR.predict(X_test)

#  prediction summary for the model
print(classification_report(y_test, y_pred_1))

# Accuracy score
LR_SC = accuracy_score(y_pred_1,y_test)
print(f"{round(LR_SC*100,2)}% Accurate")


# Here, y_test are the true class labels and y_pred are the predicted class labels in the test-set.

# In[28]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_1)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


# In[29]:


cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# In[30]:


# train a  decision tree classifier on the training set
from sklearn.tree import DecisionTreeClassifier


# instantiate the model
DT = DecisionTreeClassifier()


# fit the model
DT.fit(X_train, y_train)

#predict values using the test data
y_pred_2 = DT.predict(X_test)

#  prediction Summary by model
print(classification_report(y_test, y_pred_2))

# Accuracy score
DT_SC = accuracy_score(y_pred_2,y_test)
print(f"{round(DT_SC*100,2)}% Accurate")


# In[31]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_2)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


# In[32]:


cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# In[33]:


# train a random forest classifier on the training set
from sklearn.ensemble import RandomForestClassifier

# instantiate the model
RF = RandomForestClassifier()


# fit the model
RF.fit(X_train, y_train)

# predict values using the test data
y_pred_3 = RF.predict(X_test)

#  prediction Summary by model
print(classification_report(y_test, y_pred_3))

# Accuracy score
RF_SC = accuracy_score(y_pred_3,y_test)
print(f"{round(RF_SC*100,2)}% Accurate")


# In[34]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_3)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


# In[35]:


cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# In[36]:


# train a KNN classifier on the training set
from sklearn.neighbors import KNeighborsClassifier

# instantiate the model
KNN = KNeighborsClassifier()


# fit the model
KNN.fit(X_train, y_train)

# predict values using the test data
y_pred_4 = KNN.predict(X_test)

#  prediction Summary by model
print(classification_report(y_test, y_pred_4))

# Accuracy score
KNN_SC = accuracy_score(y_pred_4,y_test)
print(f"{round(KNN_SC*100,2)}% Accurate")


# In[37]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_4)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


# In[38]:


cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# In[39]:


# train a Linear SVM classifier on the training set
from sklearn.svm import LinearSVC

# instantiate the model
LSVC = LinearSVC(C=1, max_iter=100)


# fit the model
LSVC.fit(X_train, y_train)
# predict values using the test data
y_pred_5 = LSVC.predict(X_test)

#  prediction Summary by model
print(classification_report(y_test, y_pred_5))

# Accuracy score
LSVC_SC = accuracy_score(y_pred_5,y_test)
print(f"{round(LSVC_SC*100,2)}% Accurate")


# In[40]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_5)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


# In[41]:


cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# In[42]:


# train a GaussianNB classifier on the training set
from sklearn.naive_bayes import GaussianNB

# instantiate the model
GNB = GaussianNB()


# fit the model
GNB.fit(X_train, y_train)
# predict values using the test data
y_pred_6 = GNB.predict(X_test)

#  prediction Summary by model
print(classification_report(y_test, y_pred_6))

# Accuracy score
GNB_SC = accuracy_score(y_pred_6,y_test)
print(f"{round(GNB_SC*100,2)}% Accurate")


# In[43]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_6)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


# In[44]:


cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# In[45]:


score = [LR_SC*100,DT_SC*100,RF_SC*100,KNN_SC*100,LSVC_SC*100,GNB_SC*100]
Models = pd.DataFrame({
    'Models': ["Logistic Regression","Decision Tree","Random Forest", "KNN", "Linear SVM", "Gaussian NB"],
    'Score': score})
Models.sort_values('Score', ascending=True)


# We can see that the best performing algorithm is the RandomForest with an accuracy score of 99%

# ### ROC Curve

# In[46]:


# plot ROC Curve for the logistic regression classifier

from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_predictions(y_test, y_pred_1)

plt.title('ROC curve for Logistic regression Classifier for Predicting Fraud')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.figure(figsize=(6,4))

plt.show()


# In[47]:


# plot ROC Curve for the decision tree classifier

from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_predictions(y_test, y_pred_2)

plt.title('ROC curve for Decision tree Classifier for Predicting Fraud')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.figure(figsize=(6,4))

plt.show()


# ROC curve help us to choose a threshold level that balances sensitivity and specificity for a particular context.

# In[48]:


# plot ROC Curve for the random forest classifier

from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_predictions(y_test, y_pred_3)

plt.title('ROC curve for Random forest Classifier for Predicting Fraud')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.figure(figsize=(6,4))

plt.show()


# In[49]:


# plot ROC Curve for the KNN classifier

from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_predictions(y_test, y_pred_4)

plt.title('ROC curve for KNN Classifier for Predicting Fraud')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.figure(figsize=(6,4))

plt.show()


# In[50]:


# plot ROC Curve for the Linear SVM classifier

from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_predictions(y_test, y_pred_5)

plt.title('ROC curve for Linear SVM Classifier for Predicting Fraud')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.figure(figsize=(6,4))

plt.show()


# In[51]:


# plot ROC Curve for the gaussian naive bayes classifier

from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_predictions(y_test, y_pred_6)

plt.title('ROC curve for Gaussian Naive Bayes Classifier for Predicting Fraud')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.figure(figsize=(6,4))

plt.show()


# Interpretation :
# 1. ROC AUC is a single number summary of classifier performance. The higher the value, the better the classifier.
# 2. OC AUC of our model approaches towards 1. So, we can conclude that our classifiers does a good job in predicting whether a transcation is genuine or fraudulent.

# In[ ]:




