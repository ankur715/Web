#!/usr/bin/env python
# coding: utf-8

# In[19]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


# In[21]:


dataset = pd.read_csv('hiring.csv')


# In[23]:


dataset['experience'].fillna(0, inplace=True)
dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)


# In[28]:


X = dataset.iloc[:, :3]


# In[29]:


#Converting words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]


# In[32]:


X['experience'] = X['experience'].apply(lambda x: convert_to_int(x))


# In[34]:


y = dataset.iloc[:, -1]


# In[35]:


#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# In[36]:


#Fitting model with trainig data
regressor.fit(X, y)


# In[37]:


# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))


# In[38]:


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))

