
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('Housedata.csv', index_col=0)
df = df.replace({'yes': 1, 'no': 0})
df.head()


# In[3]:


Y = df.iloc[:, 0].values
X = df.iloc[:, 1:].values


# In[21]:


w = np.linalg.inv(np.matmul(X.T , X))  #(X.T * X)^-1


# In[22]:


y = np.matmul(X.T , Y)  #(X.T * Y)


# In[23]:


w = np.matmul(w , y)


# In[24]:


w


# In[19]:


index = 25
np.dot(w , X[index]), Y[index]


# In[36]:


Y_pred = np.array([np.dot(w, x) for x in X])


# In[37]:


Y_pred

