FROM python:3.9-slim
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY . ./
CMD gunicorn -b 0.0.0.0:8050 app:server
# if we want to have multiple workers we can use
# gunicorn --workers 4 -b 0.0.0.0:8050 app:server
#EXPOSE 8050
#CMD python3 app.py