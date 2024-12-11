#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from google.colab import drive
# drive.mount('/content/drive')


# # In[2]:


# get_ipython().run_line_magic('cd', '/content/drive/My Drive/Colab Notebooks')


# In[3]:


import pandas as pd
import numpy as np
import sklearn
import matplotlib
import keras


# In[4]:


import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


# In[5]:


# the names will be the names of each column in our pandas DataFrame
names = ['age',
 'sex',
 'cp',
 'trestbps',
 'chol',
 'fbs',
 'restecg',
 'thalach',
 'exang',
 'oldpeak',
 'slope',
 'ca',
 'thal',
 'class']

 # read the csv
cleveland=pd.read_csv("processed.cleveland.data", names=names)


# In[6]:


# It is time to print the shape of the dataframe so we can see how many examples that we have
print('Shape of DataFrame: {}'.format(cleveland.shape))
print(cleveland.loc[1])


# In[7]:


#print the data of the last 23 patietns
cleveland.loc[280:]


# In[8]:


data = cleveland[~cleveland.isin(['?'])]
data.loc[280:]


# In[9]:


data = data.dropna(axis=0)
data.loc[280:]


# In[10]:


print(data.shape)
print(data.dtypes)


# In[11]:


# convert all objects into numeric values
data = data.apply(pd.to_numeric)
data.dtypes


# In[12]:


data.describe()


# In[13]:


data.hist(figsize = (12, 12))
# plt.show()


# In[14]:


# create X and Y datasets for training
from sklearn import model_selection

X = np.array(data.drop(['class'], axis=1))
y = np.array(data['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)


# In[15]:


from keras.utils import to_categorical

Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes=None)
print(Y_train.shape)
print(Y_train[:10])


# In[16]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# define a function to build the keras model
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(5, activation='softmax'))

    # compile model
    adam = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

model = create_model()

print(model.summary())


# In[17]:


model.fit(X_train, Y_train, epochs=100, batch_size=10, verbose = 1)


# In[18]:


Y_train_binary = y_train.copy()
Y_test_binary = y_test.copy()

Y_train_binary[Y_train_binary > 0] = 1
Y_test_binary[Y_test_binary > 0] = 1
print(Y_train_binary[:20])


# In[19]:


# define a new keras model for binary classification
def create_binary_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    adam = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

binary_model = create_binary_model()

print(binary_model.summary())


# In[20]:


# fit the binary model on the training data
binary_model.fit(X_train, Y_train_binary, epochs=100, batch_size=10, verbose = 1)


# In[21]:


from sklearn.metrics import classification_report, accuracy_score
categorical_pred=model.predict(X_test)


# In[22]:


# categorical_pred


# In[23]:


categorical_pred = np.argmax(model.predict(X_test), axis=1)
print("categorical_pred", categorical_pred)


# In[24]:


# categorical_pred


# In[25]:


print('Results for Categorical Model')
print(accuracy_score(y_test, categorical_pred))
print(classification_report(y_test, categorical_pred))


# In[26]:


binary_pred = np.round(binary_model.predict(X_test)).astype(int)
print("binary_pred", binary_model.predict(X_test))


# In[27]:


print('Results for Categorical Model')
print(accuracy_score(Y_test_binary, binary_pred))
print(classification_report(Y_test_binary, binary_pred))

