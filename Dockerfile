FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies required for OpenCV and albumentations
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source code
COPY src/ src/

# Copy essential data file (geojson only)
COPY data/xView_train.geojson data/xView_train.geojson

# Copy environment variables
COPY .env .

# Declare volume mount for external data
VOLUME ["/workspace/data"]

# Default command
CMD ["/bin/bash"]
