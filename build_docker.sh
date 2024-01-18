#!/bin/bash

BASE_IMAGE_PT=nvcr.io/nvidia/pytorch:23.12-py3
BASE_IMAGE_JETSON=nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

while getopts "b:" opt; do
  case $opt in
    b)
        case $OPTARG in 
            pt)
                echo "Using PyTorch base image"
                BASE_IMAGE=$BASE_IMAGE_PT
                ;;
            jetson)
                echo "Using Jetson base image"
                BASE_IMAGE=$BASE_IMAGE_JETSON
                ;;
            *)
                echo "Invalid base image option: $OPTARG" >&2
                exit 1
                ;;
        esac
        ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done

# Build the Docker image
docker build -t flower_client:latest --build-arg BASE_IMAGE=$BASE_IMAGE .

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Docker image built successfully."
    # Run the Docker container with the --rm option to remove it after exit
    docker run --rm --network host flower_client:latest
else
  echo "Error: Docker image build failed."
fi


                
# OPENCV_DEB=OpenCV-4.5.0-aarch64.tar.gz OPENCV_URL=https://nvidia.box.com/shared/static/2hssa5g3v28ozvo3tc3qwxmn78yerca9.gz PYTORCH_URL=https://nvidia.box.com/shared/static/rehpfc4dwsxuhpv4jgqv8u6rzpgb64bq.whl PYTORCH_WHL=torch-2.0.0a0+ec3941ad.nv23.2-cp38-cp38-linux_aarch64.whl TORCHAUDIO_VERSION=v0.13.1 TORCHVISION_VERSION=v0.14.1 TORCH_CUDA_ARCH_LIST="7.2;8.7" pip3 install --no-cache-dir --verbose pycuda six