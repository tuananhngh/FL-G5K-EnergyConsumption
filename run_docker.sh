#!/bin/bash

# Build the Docker image
docker build -t flower_client:latest .

# Check if the build was successful
if [ $? -eq 0 ]; then
  echo "Docker image built successfully."
  # Run the Docker container with the --rm option to remove it after exit
  docker run --rm -it flower_client:latest
else
  echo "Error: Docker image build failed."
fi
