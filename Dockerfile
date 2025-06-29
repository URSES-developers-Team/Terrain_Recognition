# Use official PyTorch image with CUDA
FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies for OpenCV and ultralytics
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the code
COPY src/ src/
COPY data/ data/
COPY .env .
COPY data/xView_train.geojson data/

COPY data/dataset/train_images/5.tif data/dataset/train_images/
COPY data/dataset/train_images/8.tif data/dataset/train_images/
COPY data/dataset/train_images/10.tif data/dataset/train_images/

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TORCH_USE_NNPACK=0
