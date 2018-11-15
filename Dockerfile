FROM python:3.6

VOLUME ["/data"]

ENV TRAINING_CHECKPOINT_DIR=/data/training_checkpoints
ENV DB_URL=sqlite:////data/database.sqlite3
ENV VOCAB_FILE=/data/vocab.txt

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt
