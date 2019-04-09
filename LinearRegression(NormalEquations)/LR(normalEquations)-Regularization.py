
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[12]:


df = pd.read_csv('Housedata.csv', index_col=0)
df = df.replace({'yes': 1, 'no': 0})
df.insert(1,"x0",np.ones(len(df)))
df.head()


# In[24]:


Y = df.iloc[:, 0].values
X = df.iloc[:, 1:].values
lamda = 100


# In[25]:


w = np.linalg.inv(np.matmul(X.T , X) + lamda*np.identity(X.shape[1]))  #(X.T * X)^-1


# In[26]:


y = np.matmul(X.T , Y)  #(X.T * Y)


# In[27]:


w = np.matmul(w , y)


# In[28]:


index = 25
np.dot(w , X[index]), Y[index]


# In[29]:


w

