# syntax=docker/dockerfile:1

# Base image: TF 2.9.1 GPU
FROM tensorflow/tensorflow:2.9.1-gpu

# Working directory (in the container, not in host)
WORKDIR /home/app
RUN pip install --upgrade pip

# Requirements
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# OpenCV
RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y python3-opencv
RUN pip install opencv-python
