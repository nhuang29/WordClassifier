#!/bin/bash

sudo apt-get update

# Install Docker
sudo apt install docker.io

# Start Docker Service
sudo systemctl start docker
sudo systemctl enable docker

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/1.25.5/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Build and Run Docker Container
cd ~/server
sudo docker-compose up --build