# exercise

has three methods for prediction

/predict : predicts for single instance \
/predict_batch : predicts for multiple instances \
/predict_file : predicts for a file  \
\
\
\ 
to build and run docker \
build : \
docker build -t logreg_coord . \
run : \
docker run -p 8000:8000 logreg_coord
