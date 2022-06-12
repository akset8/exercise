# Exercise

### Implements the following 3 routes 
API can be tested out at :                         via swagger docs 
 - /predict - to predict for a single instance 
 - /predict_batch - to predict for multiple instances
 - /predict_file - to predict for a file with target labels, also computes accuracy metric 

Code information :

 - code.py  - file for model training 
 - inference.py - inference on test.csv with saved model
 - webserver.py - webserver implementing the above 3 routes using FastAPI
 - Dockerfile - for dockerizing the webserver
 - coords.py - class information for json parsing and validation via pydantic 
 
The Docker-image has been hosted on heroku and can be accessed via : 
