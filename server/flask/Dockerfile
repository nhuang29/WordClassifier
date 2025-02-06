# Pull tensorflow image from Python 3
FROM tensorflow/tensorflow:latest-py3
FROM python:3.10

# Setup working directory for app
WORKDIR /app

# Transfer content from current dir to /app in container
ADD . /app

# Install audio libraries
RUN apt-get update && apt-get install -y libsndfile1 libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg

RUN apt-get install -y python3

# Upgrade Pip
RUN pip install --upgrade pip

# Install python libraries
RUN pip install --force-reinstall -r requirements.txt

# Start UWSGI
CMD ["uwsgi", "app.ini"]