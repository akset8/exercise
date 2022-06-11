#!/usr/bin/env python

import uvicorn
from fastapi import FastAPI, File, UploadFile
from coords import coord, coord_list
import numpy as np
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create the app
app = FastAPI()
# load the classifier 
model=pickle.load(open("logreg_model.pickle","rb"))

@app.get('/')
def index():
    return {'message': 'Hello from Classifier!'}


#predict, prediction for single instance 
@app.post('/predict')
async def predict(data:coord):

    data = data.dict()
    x = data['x']
    y = data['y']
    inputs = [x,y]
    prediction = model.predict([inputs]).tolist()

    return {
        'prediction': prediction[0]
    }


#predict_batch, predictions for multiple examples 
#raise error if number of examples doesnt match 
@app.post('/predict_batch')
async def predict_batch(data:coord_list):

    data = data.dict()
    n = data['num_examples']
    inputs = []

    assert n == len(data['inputs'])

    for example in data['inputs']:
    	inputs.append([example['x'], example['y']])

    prediction = model.predict(inputs).tolist()

    return {
        'prediction': prediction
    }

# predict_file, predicting accuracy metric on file with target label given
@app.post('/predict_file')
async def predict_file(file: UploadFile):
	
	test = pd.read_csv(file.filename)
	
	# assert shape[1] to be 3 
	assert test.shape[1]==3

	X_test = np.array((test.iloc[:, :-1])).reshape(len(test),2)
	y_test = np.array(test.iloc[:, -1:]).reshape(len(test))
	y_pred = model.predict(X_test).tolist()
	acc = accuracy_score(y_test,y_pred)

	return {
    	'accuracy': acc,
    	'predictions':y_pred
    }


if __name__ == '__main__':
    uvicorn.run(app)