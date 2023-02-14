#!/usr/bin/env python
# coding: utf-8

# # Text Mining

# In[23]:


from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


# In[13]:


vec=CountVectorizer()


# In[14]:


corpus=["Hi my name khalid khan and i can't learing coding!!!!!"]


# In[15]:


corpus


# In[16]:


x=vec.fit_transform(corpus)


# In[17]:


x


# In[25]:


values = x.toarray()


# In[19]:


print(x.toarray())


# In[24]:


features = vec.get_feature_names()


# In[21]:


len(vec.get_feature_names())


# In[22]:


vec.vocabulary_


# In[ ]:





# In[ ]:





# In[ ]:




