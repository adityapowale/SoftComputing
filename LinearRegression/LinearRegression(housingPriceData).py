
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv('Housedata.csv',index_col=0)


# In[3]:


data.head()


# In[4]:


categorical=['driveway','recroom','fullbase','gashw','airco','prefarea']


# In[5]:


for col in categorical:
    
    temp=data.loc[:,col]
    c=temp.nunique()
    temp=temp.astype('category')
    temp.cat.categories=np.arange(c)
    
    data.loc[:,col]=temp


# In[6]:


data.head()


# In[7]:


plt.figure(figsize=(16,15))
for i in range(2,12):
    plt.subplot(4,4,i-1)
    plt.xlabel(data.columns[i])
    plt.hist(data.iloc[:,i])

plt.subplot(4,4,11)
plt.hist(data.iloc[:,1])
plt.xlabel(data.columns[1])
plt.show()


# In[8]:


plt.figure(figsize=(40,40))
for i in range(2,12):
    plt.subplot(4,4,i-1)
    plt.xlabel(data.columns[i])
    plt.scatter(data.iloc[:,0],data.iloc[:,1],c=data.iloc[:,i])
    plt.colorbar()


# In[9]:


pmax=data.loc[:,'price'].max()
pmin=data.loc[:,'price'].min()
data.loc[:,'price']=data.loc[:,'price'].apply(lambda x:(x-pmin)/(pmax-pmin))


# In[10]:


lmax=data.loc[:,'lotsize'].max()
lmin=data.loc[:,'lotsize'].min()
data.loc[:,'lotsize']=data.loc[:,'lotsize'].apply(lambda x:(x-lmin)/(lmax-lmin))


# In[11]:


data.head(1)


# In[12]:


dataset=data.values


# In[13]:


split_ratio=0.8
tot_data=dataset.shape[0]


# In[14]:


xtrain=dataset[:int(tot_data*split_ratio),1:]
ytrain=dataset[:int(tot_data*split_ratio),:1]

xtest=dataset[int(tot_data*split_ratio):,1:]
ytest=dataset[int(tot_data*split_ratio):,:1]


# In[15]:


xtrain.shape,ytrain.shape,xtest.shape,ytest.shape


# In[16]:


inputs=xtrain.shape[1]
outputs=ytrain.shape[1]
learning_rate=0.01
x_axis = []
y_axis = []


# In[17]:


w=np.random.random(size=(inputs,outputs))
b=np.random.random(size=(outputs,))
w.shape,b.shape


# In[18]:


def forward(x):
    return np.dot(x,w)+b


# In[19]:


def diff(y,y_):
    out=(y_- y)
    return out


# In[20]:


def mse(y,y_):
    return np.mean(np.multiply(y-y_ , y-y_))


# In[21]:


def train():
    warr=[]
    for epoch in range(1000):

        ypred=forward(xtrain)
        l=diff(ytrain,ypred).reshape(-1)
        y_axis.append((l.mean())**2)
        x_axis.append(epoch)
        for i in range(inputs):
            dw=np.mean(np.multiply(l,xtrain[:,i]))
            w[i]=w[i]- learning_rate*(dw)
        
        b[0] = b[0] - learning_rate*np.mean(l)
        
        if(epoch%10==0):
            print("Epoch : {}".format(epoch))

train()


# In[33]:


def denorm(y):
    return (y*(pmax-pmin))+pmin


def test():
    ypred=forward(xtest)
    index=20
    print(denorm(ypred[index]),denorm(ytest[index]))
    e = mse(ytest,ypred)
    print(e)
test()


# In[29]:


plt.plot(x_axis , y_axis)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()
