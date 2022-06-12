FROM continuumio/anaconda3:4.4.0
COPY . /usr/app/
EXPOSE 8000
WORKDIR /usr/app/
RUN pip install -r requirements.txt
CMD ["uvicorn", "webserver:app","--reload","--host", "0.0.0.0",,"--port","8000"]