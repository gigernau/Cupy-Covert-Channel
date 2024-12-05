# Cupy-Covert-Channel
<img src=img/Covert_communication.png align="center">

## Overview
This project explores the creation of covert communication channels using GPU metrics. By leveraging GPU operations to modulate resource usage (e.g., power, temperature, utilization), messages can be encoded and transmitted between a sender and receiver. The focus is on analyzing the feasibility, bandwidth, and detectability of these channels, with applications in security research and understanding GPU-based vulnerabilities. The repository includes code for classification models, and testing frameworks for covert channel experiments.

# Setup for Ubuntu Linux OS

## 1) Update : 
	sudo apt update
  
## 2) Install dependencies:
    sudo apt install build-essential 

## 3) Install pip3 for Python3: 
	sudo apt install python3-pip  && python3 -m pip install --upgrade pip

## 4) Install Python3 modules: 
	python3 -m pip install -r requirements.txt
	
## 5) Check Cuda version or Install [CUDA](https://developer.nvidia.com/cuda-toolkit)
	nvcc -V

## 6) Install CuPy module ( e.g., for CUDA 12.1 ):
	python3 -m pip install cupy-cuda12x

## 7) Unzip dataset:
    unzip dataset/sort_transpose.zip


# Usage Example
## TRAINING MODEL

    python training.py --split 10 --data dataset/sort_transpose/ --op transpose sort


## SENDER OF SECRET MESSAGE of COVERT CHANNEL

    python sender.py --op transpose sort --model model/\[\'transpose\'\,\ \'sort\'\]_1.0000.joblib --message 1010




## RECEIVER OF SECRET MESSAGE

    python receiver.py --folder message --op transpose sort --model model/\[\'transpose\'\,\ \'sort\'\]_1.0000.joblib --test 1010


<!--- [![Say Thanks!](https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg)](https://saythanks.io/to/gianluca.delucia) -->
