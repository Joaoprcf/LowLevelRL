# Use an image that includes CUDA, such as from NVIDIA
FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    g++ \
    make

# Copy your project files into the Docker image
WORKDIR /project
COPY . .

# Build and run tests
CMD ["make", "fast_test"]
