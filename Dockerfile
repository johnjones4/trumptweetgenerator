FROM python:3.6

VOLUME ["/data"]

ENV DB_URL=sqlite:////data/database.sqlite3

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt
