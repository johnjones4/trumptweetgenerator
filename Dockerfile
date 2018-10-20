FROM python:3.6

VOLUME ["/data"]

ENV TRAINING_CHECKPOINT_DIR=/data/training_checkpoints
ENV DB_URL=sqlite:////data/database.sqlite3
ENV MODEL_WEIGHTS_FILE=/data/model_weights.h5
ENV MODEL_ARCH_FILE=/data/model_architecture.json

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt
