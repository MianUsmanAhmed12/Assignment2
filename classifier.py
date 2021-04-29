#!/usr/bin/env python
# coding: utf-8

# In[2]:


''' Import the dataset.csv file and run the code in python IDE'''
#import the necessary libraries
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer


# In[3]:


#read the dataset file
dataset = pd.read_csv('dataset.csv')


# In[4]:


#remove unuseful columns, These columns were filled with nan values and some we do not need becasue of the problem statement we formulated 
dataset = dataset.drop(['Unnamed: 0', 'Date', 'Outcome', 'Outcome linked to object of search', 'Removal of more than just outer clothing', 'Longitude', 'Latitude'], axis = 1)

dataset['Bame'] = dataset['Bame']*1
dataset['noaction'] = dataset['noaction']*1


# In[5]:


#noaction is the output value we are going to predict, so we would separate it from input features.
X = dataset.drop(['noaction'], axis = 1)


# In[6]:


# the noaction series is saved into output variable
y = dataset.noaction


# In[7]:


#fill the nan values with mode, The mode is the value that appears most often in a set of data values. 
#dataset['Gender'] = dataset['Gender'].fillna(dataset['Gender'].mode()[0])
#dataset['Age range'] = dataset['Age range'].fillna(dataset['Age range'].mode()[0])
#dataset['Self-defined ethnicity'] = dataset['Self-defined ethnicity'].fillna(dataset['Self-defined ethnicity'].mode()[0])
#dataset['Officer-defined ethnicity'] = dataset['Officer-defined ethnicity'].fillna(dataset['Officer-defined ethnicity'].mode()[0])
#dataset['Object of search'] = dataset['Object of search'].fillna(dataset['Object of search'].mode()[0])
imp_mean = SimpleImputer(missing_values=np.nan,  strategy='constant', fill_value=X.mode())
X = imp_mean.fit_transform(X)


# In[8]:


#Most of the features are categorical so we will encode them to be fed into the model. Here Ordinal Encoder from sklearn library has been used. It encodes categorical features as an integer array.
oe = OrdinalEncoder()
X = oe.fit_transform(X)


# In[9]:


#Now we would Standardize features by removing the mean and scaling to unit variance. Sklearn provide StandardScalar library from preprocessing package
scalar = StandardScaler()
X = scalar.fit_transform(X)


# In[10]:


#the dataset is split into train and test set using StratifiedShuffleSplit which is library for sklearn. It returns stratified randomized folds. The folds are made by preserving the percentage of samples for each class.
sss = StratifiedShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# In[11]:


#We would use Random Forest Classifier for classification. A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
#n_estimators = 500
#max_depth=500
#random_state=0
#max_samples = 500
classifier = RandomForestClassifier( )
classifier.fit(X_train, y_train)


# In[12]:


#predict the test set
yhat = classifier.predict(X_test)


# In[13]:


#check the accuracy of model using different classification metric
print(f1_score(y_test, yhat, average="macro"))
print(recall_score(y_test, yhat, average="macro"))    
print(accuracy_score(y_test, yhat))


# In[14]:


#We will also use confusion matrix to visulaize the performance. Confusion matrix also known as an error matrix, is a specific table layout that allows visualization of the performance of an algorithm.
cf_matrix = confusion_matrix(y_test, yhat, normalize='true')
plt.figure(figsize = (10,7))
sns.heatmap(cf_matrix, annot=True, xticklabels = sorted(set(y_test)), yticklabels = sorted(set(y_test)),cbar=False)
plt.title('Normalized Confusion Matrix', fontsize = 23)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()


# In[15]:


#we would get the feature importances to see the contribution of different features to machine learning model.
importances = list(zip(classifier.feature_importances_, dataset.columns))
importances.sort(reverse=True)

print(importances)


# In[16]:


#we would generate the barplot of feature importances to get better understanding of scores.
pd.DataFrame(importances, index=[x for (_,x) in importances]).plot(kind = 'bar')


# In[ ]:




