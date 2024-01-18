# syntax=docker/dockerfile:1
ARG _DEV_CONTAINERS_BASE_IMAGE=tensorflow-gpu

# FROM python:latest
FROM tensorflow/tensorflow:latest-gpu

# USER root 

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_NAME=signal_ai
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd ${USER_NAME} --gid ${USER_GID}\
    && useradd -l -m ${USER_NAME} -u ${USER_UID} -g ${USER_GID} -s /bin/bash
   #  && usermod -aG sudo ${USER_NAME}

RUN apt update \
   && apt -y install --no-install-recommends \
   git python3-tk iproute2 sudo x11-apps
#     libfftw3-3 libhdf5-dev libnetcdf-dev libfftw3-dev \
#     # python3-numpy

RUN pip install --upgrade pip

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && \
   rm /tmp/requirements.txt

# ENV USER=${USER_NAME}

RUN echo "signal_ai ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER_NAME}
RUN chmod 0440 /etc/sudoers.d/${USER_NAME}

# RUN chown -R ${USER_NAME}:${USER_NAME} /${USER_NAME}
    
# COPY requirements-dev.txt /tmp/requirements-dev.txt
# RUN pip install -r /tmp/requirements-dev.txt && \
#     rm /tmp/requirements-dev.txt 

USER ${USER_NAME}

# # Switch to user
# USER ${uid}:${gid}