FROM python:3.9-slim
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN pip install -r requirements_test.txt
EXPOSE 5000
COPY . ./
CMD gunicorn -b 0.0.0.0:5000 app:server
# if we want to have multiple workers we can use
# gunicorn --workers 4 -b 0.0.0.0:8050 app:server

#CMD python3 app.py