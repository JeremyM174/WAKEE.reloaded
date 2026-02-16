FROM apache/airflow:2.11.0-python3.10

USER root

RUN apt-get update \
    && apt-get install -y git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt .

RUN pip install -r requirements.txt
