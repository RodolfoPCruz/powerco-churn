#!/bin/bash

# Stop and remove any existing container with this name
docker rm -f powerco_container 2>/dev/null || true

# Run the container
docker run -it -d --rm \
    --name powerco_container \
    -w /app \
    -p 8888:8888 \
    -v "$(pwd)":/app \
    powerco

