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

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy only necessary source files
COPY src/ src/

# Copy only essential data (geojson) - this is small
COPY data/xView_train.geojson data/xView_train.geojson

# Copy environment config
COPY .env .

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TORCH_USE_NNPACK=0

# Create volume mount points
VOLUME ["/workspace/data"]

CMD ["/bin/bash"]
