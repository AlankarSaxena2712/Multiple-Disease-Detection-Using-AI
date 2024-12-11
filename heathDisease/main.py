import pandas as pd
import numpy as np
from sklearn import model_selection

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

data = cleveland[~cleveland.isin(['?'])]
data = data.dropna(axis=0)
data = data.apply(pd.to_numeric)

X = np.array(data.drop(['class'], axis=1))
y = np.array(data['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

from keras.utils import to_categorical

Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes=None)

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

model.fit(X_train, Y_train, epochs=100, batch_size=10, verbose = 1)
model.save('heart_disease_categorial_model.h5')

Y_train_binary = y_train.copy()
Y_test_binary = y_test.copy()

Y_train_binary[Y_train_binary > 0] = 1
Y_test_binary[Y_test_binary > 0] = 1

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

binary_model.fit(X_train, Y_train_binary, epochs=100, batch_size=10, verbose = 1)
binary_model.save('heart_disease_binary_model.h5')
print("XTest", X_test)

categorical_pred=model.predict(X_test)
categorical_pred =np.argmax(model.predict(X_test), axis=1)

print("categorical_pred", categorical_pred)

# binary_pred = np.round(binary_model.predict(X_test)).astype(int)
print("binary_pred", binary_model.predict(X_test))


# This is the AI model code I have written and it shows the accuracy of the training and testing dataset and now I want to input the data from the user to predict the outcome

