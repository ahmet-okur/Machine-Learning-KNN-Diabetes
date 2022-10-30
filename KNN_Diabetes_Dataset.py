#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[4]:


diabetes_df = pd.read_csv("/Users/ahmetokur/Desktop/Datasets/diabetes.csv")


# In[5]:


diabetes_df


# In[7]:


diabetes_df.isnull().sum()


# ## Splitting data

# In[8]:


features = ['Pregnancies', 'Glucose', 'BloodPressure', 
            'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
x= diabetes_df[features]
y = diabetes_df.label


# ## Train and Test

# In[9]:


from sklearn.model_selection import train_test_split


# In[99]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)


# In[100]:


from sklearn.neighbors import KNeighborsClassifier


# In[101]:


knn_clsf = KNeighborsClassifier(n_neighbors=16)


# In[102]:


knn_clsf.fit(x_train, y_train)


# In[103]:


knn_clsf.score(x_test, y_test)


# ## Evaluating the Model

# In[104]:


y_pred = knn_clsf.predict(x_test)


# In[105]:


#ACCURACY
from sklearn.metrics import accuracy_score
accuracy_score(y_pred, y_test)


# In[109]:


#precision
from sklearn.metrics import precision_score
precision_score(y_pred, y_test, average=None) # average=None


# In[107]:


# Recall
from sklearn.metrics import recall_score
recall_score(y_pred, y_test)


# In[108]:


#f1 score
from sklearn.metrics import f1_score
f1_score(y_pred, y_test)


# In[110]:


pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['prediction'], margins=True)


# In[ ]:




