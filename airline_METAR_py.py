#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

#from sklearn import preprocessing
#from sklearn import utils


# In[3]:


#reading CSV file downloaded from ICAO

metar=pd.read_csv('Weather Condition Current METAR.csv')


# In[4]:


#reading list of columns
metar.columns


# In[5]:


#taking count of rows inside file

metar.count()


# In[6]:


#Checking df from the top

metar.head()


# In[7]:


#checking seaborn pairplot to observe if its look like NON-LINEAR

sns.pairplot(metar)


# In[8]:


#taking mean value of dangerous column to ascertain safe or unsafe
#1 means unsafe
#0 means safe

mean =metar["dangerous"].mean()
mean
metar['dangerous'] = np.where(metar['dangerous'] >= mean, 1,0)


# In[9]:


#takiing mean value
mean


# In[10]:


#checking changed METAR Dataframe

metar.head()


# In[11]:


#creating another df named factor with relevant columns for processing

factor= metar.filter(['visibility', 'wind', 'precipitation', 'freezing', 'VMC_IMC'],axis=1)


# In[12]:


#cheking if its showing right ones

factor.head()


# In[13]:


#Changing data type of all columns to float64


metar['dangerous'] = metar['dangerous'].astype(np.float64)
metar['visibility'] = metar['visibility'].astype(np.float64)
metar['wind'] = metar['wind'].astype(np.float64)
metar['VMC_IMC'] = metar['VMC_IMC'].astype(np.float64)


# In[14]:


metar.info()


# In[15]:


#separating train and test data set with test set as 30%

X = factor
y = metar['dangerous']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[16]:


model = SVC(gamma='auto', kernel='sigmoid')


# In[17]:


#fitting model

model.fit(X_train, y_train)


# In[18]:


predictions = model.predict(X_test)


# In[19]:


print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


# In[20]:


#As we are getting class 1.0 as 0.00 under all of the Precision, Recall, and F1
#We are finding better parameters for C and Gamma values


# In[21]:


param_grid = {'C':[0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001]}


# In[22]:


#using GRidSEarchCV to find better values of C and Gamma

grid = GridSearchCV(SVC(),param_grid,verbose=4)


# In[23]:


grid.fit(X_train,y_train)


# In[24]:


#printing best values for C and Gamma
grid.best_params_


# In[25]:


grid.best_estimator_


# In[26]:


grid_predictions = grid.predict(X_test)


# In[27]:


print(confusion_matrix(y_test,grid_predictions))
print('\n')
print(classification_report(y_test,grid_predictions))


# In[28]:


#Still getting the same result, so sticking with it

