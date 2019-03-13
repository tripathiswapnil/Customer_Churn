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
from keras.layers import Dropout

#Initialising the ANN 
classifier = Sequential()

#Adding the input layer and the first hidden layer with dropout
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim=11)) #taking average for dense output 11+1/2 =6
#in above we are providing input_dim because it is the first hidden layer $ does not know what to expect 
classifier.add(Dropout(p=0.1))
#Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu')) #relu is rectifier activation function
classifier.add(Dropout(p=0.1))
#Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) #output_dim is 1 because in output layer there is only 1 node
#relu is replaced by sigmoid activation funciton 
#softmax is used as activation when there are more than 2 categories, ie output is more than 2 
 classifier.add(Dropout(p=0.1))
#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

#PART 3 - MAKING PREDICTIONS AND EVALUATING MODEL

# Predicting the Test set results
y_pred = classifier.predict(X_test) #for confusion matrix, threshold is used with predict
y_pred = (y_pred > 0.5) #if y-pred > then true, if not then false 

#Predicting a single new obersvation
"""Predict if the customer with following information will leave the bank:
    Geography: France
    Credit Score: 600
    Gender: Male
    Age: 40
    Tenure: 3
    Balance: 60000
    Number of products: 2
    Has credit card: Yes
    Is Active member: Yes
    Estimated Salary: 50000"""
#new_prediction = classifier.predict(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])) 
#encoding the data with the help of dataset to check and verify encoding in above 
#now since the model was trained on X_Train and X_train was scaled(stadarisation),
#new prediction has to be made on same scale in Line 27. We take 'sc' object
new_prediction = classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]]))) 
new_prediction = (new_prediction > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#PART 4 -EVALUATING, IMPROVING AND TUNING THE ANN

#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim=11)) #taking average for dense output 11+1/2 =6
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu')) #relu is rectifier activation function
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) #output_dim is 1 because in output layer there is only 1 node
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier 
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1) 
mean = accuracies.mean()
variance = accuracies.std()

#Improving the ANN
#Dropout Regularization to reduce overfitting if needed 


#Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim=11)) #taking average for dense output 11+1/2 =6
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu')) #relu is rectifier activation function
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) #output_dim is 1 because in output layer there is only 1 node
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier 
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_










