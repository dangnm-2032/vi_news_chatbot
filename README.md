# Vietnamese News Summary Bot

## Introduction:
Are you interested in something that happened recently but don't have time to read all the related articles to fully understand the situation? This open source project is for you!

You just need to enter keywords about the event you want to learn about, the system will automatically search, filter and summarize for you.

Below is the project demo.
![Project Demo](https://github.com/dangnm-2032/vi_news_chatbot/blob/main/metadata/demo.png)

## Features:
- Vietnamese News Summary Bot
- Query news from VNE
- Perform Silhoutte and Kmeans to filter most related news
- Nvidia GPU accessibility

## How to use
### Install Docker
> Please follow the instruction of Docker: [Install Docker Engine](https://docs.docker.com/engine/install/)
> 
> **IMPORTANT FOR LINUX** Do Docker post installation for linux

### Install Nvidia GPU driver
```
sudo add-apt-repository ppa:graphics-drivers/ppa  
sudo apt update  
sudo apt install ubuntu-drivers-common  
sudo apt dist-upgrade  
sudo reboot  
sudo ubuntu-drivers autoinstall  
sudo reboot
```
Test the installation, run:
```
nvidia-smi
```

### Installing Nvidia Container Toolkit
```
distribution=$(. /etc/os-release;echo  $ID$VERSION_ID)  
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -  
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Install application
```
docker compose build
```

### Configurration
> Please take a look in to config.yaml

### Run
```
docker compose up
```

### Stop
```
docker compose down
```

        