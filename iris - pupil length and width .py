#!/usr/bin/env python
# coding: utf-8

# In[134]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[128]:


iris=sns.load_dataset('iris')


# In[132]:


iris=iris[['petal_length','petal_width']]


# In[133]:


iris


# In[137]:


x=iris['petal_length']
y=iris['petal_width']


# In[141]:


plt.scatter(x,y,c='r')
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.show()


# In[142]:


from sklearn.model_selection import train_test_split


# In[145]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)


# In[165]:





# In[153]:


import numpy as np


# In[156]:


x_train=np.array(x_train)
y_train=np.array(y_train)


# In[203]:


w=0.1
b=0.1


# In[204]:


def cost_function(x,y,w,b):

    cost=0
    f_x=w*x+b
    temp=(f_x-y)**2
    cost+=temp
    cost=(1/(2*m))*cost
    return cost


# In[205]:


m=len(x_train)


# In[206]:


cost=np.zeros(m)
for i in range(m):
    cost[i]=cost_function(x_train[i],y_train[i],w,b)


# In[207]:


def predict(x,w,b):
    m=len(x)
    f_x=np.zeros(m)
    for i in range(m):
        f_x[i]=w*x[i]+b
    return f_x


# In[208]:


w=0.1
b=0.1
p=predict(x_train,w,b)


# In[209]:


p


# In[216]:


plt.plot(x_train,p,c='b')
plt.scatter(x_train,y_train,c='r')
plt.plot(x_train,cost)
plt.show()


# In[221]:


from mpl_toolkits.mplot3d import Axes3D
# Create a new figure for the 3D plot
fig = plt.figure()

# Add a 3D subplot
ax = fig.add_subplot(111, projection='3d')

# Create a scatter plot
ax.scatter(x_train, y_train, cost)


# In[223]:


def partial(x,y,w,b):
    sw=0
    sb=0
    m=len(x)
    for i in range(m):
        f_x=w*x[i]+b
        tw=(f_x-y[i])*x[i]
        tb=(f_x-y[i])
        sw+=tw
        sb+=tb
    sw=(1/m)*sw
    sb=(1/m)*sb
    return sw,sb


# In[234]:


def final_val(x,y,w_in,b_in,alpha,iteration,partial):
    w=w_in
    b=b_in
    for i in range(iteration):
        dw,db=partial(x,y,w,b)
        w=w-alpha*dw
        b=b-alpha*db
    return w,b


# In[243]:


w_in=0.1
b_in=0.1
iteration=10000
alpha=0.001
w,b=final_val(x_train,y_train,w_in,b_in,alpha,iteration,partial)


# In[244]:


w


# In[245]:


b


# In[246]:


p=predict(x_train,w,b)


# In[247]:


plt.plot(x_train,p,c='b')
plt.scatter(x_train,y_train,c='r')
plt.show()


# In[248]:


print(f"w={w}and b={b}")


# now predicting the result for test values

# In[252]:


x_test=np.array(x_test)
y_test=np.array(y_test)


# In[254]:


p=prediction(x_test,w,b)


# In[258]:


plt.plot(x_test,p,c='b')
plt.scatter(x_test,y_test,c='r',marker='x')
plt.show()


# In[ ]:




