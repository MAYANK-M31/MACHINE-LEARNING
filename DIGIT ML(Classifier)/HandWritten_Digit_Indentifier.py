#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
data = fetch_openml("mnist_784", version=1)


# In[2]:


digit_data = data


# In[3]:


x, y = digit_data["data"], digit_data["target"]


# In[48]:


import numpy as np 

x_train, y_train = x[:6000], y[:60000]
x_test , y_test = x[6000:], y[6000:]


# In[49]:


shuffled = np.random.permutation(6000)
x_train,y_train = x_train[shuffled], y_train[shuffled]


# In[69]:


y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
y_train2 = (y_train==1)
y_test2 = (y_test==1)


# In[70]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(tol = 0.1)
model.fit(x_train,y_train2)


# In[68]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
some_digit = x_train[5]
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image)


# In[71]:


predicted = model.predict([some_digit])
predicted


# In[79]:


from sklearn.model_selection import cross_val_score
err = cross_val_score(model,x_train,y_train2,cv=3,scoring="accuracy")


# In[80]:


err.mean()*100 


# In[ ]:




