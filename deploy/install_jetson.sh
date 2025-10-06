#!/bin/bash

# Installation script for NVIDIA Jetson devices
# Tested on Jetson Nano, Xavier NX, AGX Xavier

set -e

echo "========================================="
echo "Guardian Angel - Jetson Installation"
echo "========================================="

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "Warning: This script is designed for NVIDIA Jetson devices"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install dependencies
echo "Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    libopencv-dev \
    libopencv-python \
    libhdf5-serial-dev \
    hdf5-tools \
    libhdf5-dev \
    zlib1g-dev \
    zip \
    libjpeg8-dev \
    liblapack-dev \
    libblas-dev \
    gfortran

# Install PyTorch (pre-built for Jetson)
echo "Installing PyTorch for Jetson..."
wget https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i0q926hy4imzs2ph.whl -O torch-2.0.0-cp38-cp38-linux_aarch64.whl
pip3 install torch-2.0.0-cp38-cp38-linux_aarch64.whl
rm torch-2.0.0-cp38-cp38-linux_aarch64.whl

# Install torchvision (build from source for compatibility)
echo "Installing torchvision..."
git clone --branch v0.15.0 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.15.0
python3 setup.py install --user
cd ..
rm -rf torchvision

# Install project requirements
echo "Installing Guardian Angel requirements..."
pip3 install -r requirements.txt

# Download YOLOv8-pose model
echo "Downloading YOLOv8-pose model..."
mkdir -p models
python3 -c "from ultralytics import YOLO; model = YOLO('yolov8n-pose.pt')"

# Create necessary directories
echo "Creating directories..."
mkdir -p data/video_clips
mkdir -p logs

# Set permissions
chmod +x deploy/install_jetson.sh

echo "========================================="
echo "Installation complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Edit configs/system_config.yaml with your camera settings"
echo "2. Run: python3 src/main.py"
echo ""
echo "For Docker deployment:"
echo "1. cd deploy/docker"
echo "2. docker-compose up -d"
