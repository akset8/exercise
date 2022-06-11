#!/usr/bin/env python
import numpy as np
import pandas as pd
import pickle 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

filename = 'logreg_model.pickle'
model = pickle.load(open(filename, 'rb'))

# Model Inference 
# metric chosen is accuracy since the dataset is well balanced 
# test trained model on accuracy metric  
test = pd.read_csv('test.csv')
X_test = np.array((test.iloc[:, :-1])).reshape(len(test),2)
y_test = np.array(test.iloc[:, -1:]).reshape(len(test))
y_pred = model.predict(X_test)

print ("Accuracy on Test data ", accuracy_score(y_test,y_pred))
print ("predictions are ", y_pred)
