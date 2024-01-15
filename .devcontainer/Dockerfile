# syntax=docker/dockerfile:1
ARG _DEV_CONTAINERS_BASE_IMAGE=tensorflow-gpu

# FROM python:latest
FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

RUN apt update \
   && apt -y install --no-install-recommends \
   git python3-tk iproute2
#     libfftw3-3 libhdf5-dev libnetcdf-dev libfftw3-dev \
#     # python3-numpy

# RUN pip install --upgrade pip

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && \
   rm /tmp/requirements.txt
    
# COPY requirements-dev.txt /tmp/requirements-dev.txt
# RUN pip install -r /tmp/requirements-dev.txt && \
#     rm /tmp/requirements-dev.txt    