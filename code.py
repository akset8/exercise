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

