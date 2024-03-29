#!/usr/bin/env python
# coding: utf-8

# ## Credit Fraud Detection

# Objective: To build a machine learning model to identify fraudulent credit card transactions.

# ### Steps Taken

# 1. Data Collection and preprocessing
# 2. Exploratory Data Analysis
# 3. Model Implementation

# ### 1. Data Collection and preprocessing

# Data used for project: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# 
# About data: The dataset contains transactions made by credit cards in September 2013 by European cardholders. 
# This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
# 
# It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

# In[2]:


pip install -U imbalanced-learn


# In[3]:


pip install xgboost


# In[5]:


# import necessary libraries
# importing necessary libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

# importing models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from xgboost import XGBClassifier

# libaries for under sampling 
from imblearn.under_sampling import RandomUnderSampler


# importing evaluation metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score



from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib


# In[6]:


# load dataset
credit_card_df=pd.read_csv('creditcard.csv')
credit_card_df.head()


# In[7]:


# check the size of the dataset
credit_card_df.shape


# In[8]:


# get the  dataset info
credit_card_df.info()


# In[9]:


# check for null values in the data
credit_card_df.isna().sum()


# In[10]:


# check for any duplicate data in the dataset
credit_card_df.duplicated().any()


# In[11]:


# drop duplicates data
credit_card_df=credit_card_df.drop_duplicates()


# In[12]:


credit_card_df['Amount']


# In[13]:


# Normalising the data
# All the feature from v1-v28 are already in normalised form only amount needs to be normalised
# using StandardScaler to normalise the amount feature

scaler=StandardScaler()
credit_card_df['Amount']=scaler.fit_transform(pd.DataFrame(credit_card_df['Amount']))


# In[14]:


credit_card_df['Amount']


# ### 2. Exploratory Data Analysis

# In[15]:


# Correlation heatmap
sns.heatmap(credit_card_df.corr(), cmap='YlGnBu', annot=False)


# In[16]:


# let's check for traget variable
credit_card_df['Class'].value_counts()


# In[17]:


credit_card_df['Class'].value_counts().plot(kind='bar', color=['red','blue'])


# It seems there is a class imbalance problem in the target variable where the fraud which is 1 which is very low in number as compare to non fraudulent transaction. To solve the class imbalance problem I am going to use random Under-sampling which is removing some data from non-fraudulent util it is balance with the fraudulent transaction data.

# ### Random Under Sampling

# In[18]:


# splitting the data into X and y
X=credit_card_df.drop('Class', axis=1)
y=credit_card_df['Class']


# In[19]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[20]:


# random under-sampling
rus=RandomUnderSampler(random_state=42, replacement=True)
x_rus,y_rus=rus.fit_resample(X_train,y_train)

x_rus.shape


# In[21]:


# Visualize class distribution before and after sampling
plt.figure(figsize=(12, 6))

# Plot original class distribution
plt.subplot(1, 2, 1)
sns.countplot(x=y)
plt.title("Original Class Distribution")

# Plot class distribution after SMOTE
plt.subplot(1, 2, 2)
sns.countplot(x=y_rus)
plt.title("Class Distribution After Under-sampling")

plt.tight_layout()
plt.show()


# ### 3. Model Implementation

# In[22]:


models={'Logistic Regression':LogisticRegression(),
        'Random Forest Classifier':RandomForestClassifier(),
       'Support Vector Machine':SVC(),
       'XGBoost':XGBClassifier()}


# In[23]:


models


# In[24]:


# create a function to fit and score models
def fit_and_score(models, X_train,X_test, y_train, y_test):
    # set random seed
    np.random.seed(42)
    #make a dictionary to keep model scores
    model_scores=[]
    #Loop through models
    for name,model in models.items():
        # Fit the model
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        
#         Plotting confusion matrix
        labels=['Normal', 'Fraud']
        conf_matrix=confusion_matrix(y_test,y_pred)
        
        plt.figure(figsize=(12, 12))
        sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d", cmap="Oranges")  # You can change "Blues" to another colormap
        plt.title(f"Confusion Matrix for {name}")
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        plt.show()
        
        #Evaluate the model and append its score
        print(f"Evaluating {name}....")
        report_dict=classification_report(y_test,y_pred, output_dict=True)
        
                # Extract relevant metrics from the classification report
        precision_0 = report_dict['0']['precision']
        recall_0 = report_dict['0']['recall']
        f1_0 = report_dict['0']['f1-score']
        
        precision_1 = report_dict['1']['precision']
        recall_1 = report_dict['1']['recall']
        f1_1 = report_dict['1']['f1-score']

        model_scores.append({
            'Model': name,
            'Precision_0': precision_0,
            'Recall_0': recall_0,
            'F1_0': f1_0,
            'Precision_1': precision_1,
            'Recall_1': recall_1,
            'F1_1': f1_1
        })
        
    model_scores=pd.DataFrame(model_scores)
    return model_scores


# In[25]:


model_scores=fit_and_score(models=models, X_train=x_rus,X_test=X_test,y_train=y_rus ,y_test=y_test)
model_scores


# In[26]:


model_scores


# In[27]:


# plot the results
# Melt the DataFrame to make it suitable for plotting
df_melted = pd.melt(model_scores, id_vars=['Model'], var_name='Metric', value_name='Score')

# Plotting using seaborn
plt.figure(figsize=(14, 8))
sns.barplot(x='Model', y='Score', hue='Metric', data=df_melted, palette='viridis')
plt.title('Model Comparison - Precision, Recall, and F1-Score')
plt.xlabel('Model')
plt.ylabel('Score')
plt.show()


# In[ ]:




