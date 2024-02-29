# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/engine/reference/builder/

# ARG PYTHON_VERSION=3.11.6
# FROM python:${PYTHON_VERSION}-slim as base
FROM nvidia/cuda:11.6.1-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Copy the source code into the container.
COPY . .

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN apt update && \
    apt upgrade -y && \
    apt-get install -y \
        openjdk-17-jdk\
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements.txt
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install --upgrade numpy
RUN pip3 install --upgrade wandb

# Switch to the non-privileged user to run the application.
# USER appuser

# Expose the port that the application listens on.
EXPOSE 7860

# Run the application.
# USER appuser
CMD python3 main.py