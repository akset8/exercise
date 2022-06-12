#!/usr/bin/env python
import numpy as np
import pandas as pd
import pickle 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Read train dataset
train = pd.read_csv('train.csv')
X_train = np.array((train.iloc[:, :-1])).reshape(len(train),2)
y_train = np.array(train.iloc[:, -1:]).reshape(len(train))

# Model Training 
# Fit logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

# save model 
filename = 'logreg_model.pickle'
pickle.dump(model, open(filename, 'wb'))

# Model Inference 
# metric chosen is accuracy since the dataset is well balanced 
# test trained model on accuracy metric  
test = pd.read_csv('test.csv')
X_test = np.array((test.iloc[:, :-1])).reshape(len(test),2)
y_test = np.array(test.iloc[:, -1:]).reshape(len(test))
y_pred = model.predict(X_test)
#print (y_pred.shape, y_test.shape)
print ("Accuracy on Train Data ", accuracy_score(y_train,model.predict(X_train)))
print ("Accuracy on Test data ", accuracy_score(y_test,model.predict(X_test)))

