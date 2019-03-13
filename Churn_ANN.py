#Part 1 - Data Preprocessing 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values #range of 3 to 12 but 12 is excluded so 13
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder() #1st encoder
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) #index 1 is country, index 0 is credit score 
labelencoder_X_2 = LabelEncoder() #2st encoder
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1]) #dummy variable creating / avoiding categorical ordering
X = onehotencoder.fit_transform(X).toarray() 
X = X[:, 1:] #avoiding dummy variable trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#PART 2 -MAKING THE REAL ANN 

#import Keras lib
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising the ANN 
classifier = Sequential()

#Adding the input layer and the first hidden layer 
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim=11)) #taking average for dense output 11+1/2 =6
#in above we are providing input_dim because it is the first hidden layer $ does not know what to expect 

#Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu')) #relu is rectifier activation function

#Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid')) #output_dim is 1 because in output layer there is only 1 node
#relu is replaced by sigmoid activation funciton 
#softmax is used as activation when there are more than 2 categories, ie output is more than 2 
 
#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

#PART 3 - MAKING PREDICTIONS AND EVALUATING MODEL

# Predicting the Test set results
y_pred = classifier.predict(X_test) #for confusion matrix, threshold is used with predict
y_pred = (y_pred > 0.5) #if y-pred > then true, if not then false 
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)