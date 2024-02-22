# syntax=docker/dockerfile:1.2
FROM python:3.10.9
# put you docker configuration here

WORKDIR /app

COPY challenge /app/challenge
COPY data /app/data
COPY requirements.txt /app/

RUN pip install -r requirements.txt

CMD uvicorn challenge.api:app --port=8000 --host=0.0.0.0