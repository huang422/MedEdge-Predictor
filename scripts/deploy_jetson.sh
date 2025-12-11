#!/bin/bash
# =============================================================================
# Deployment Script for NVIDIA Jetson Orin
# =============================================================================

set -e

# Configuration
IMAGE_NAME="medical-prediction-dashboard"
CONTAINER_NAME="medical-dashboard"
TAR_FILE="${IMAGE_NAME}.tar"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Jetson Orin Deployment Script${NC}"
echo -e "${GREEN}========================================${NC}"

# Function to show usage
usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build     - Build Docker image for ARM64"
    echo "  export    - Export image to tar file for transfer"
    echo "  load      - Load image from tar file"
    echo "  start     - Start the dashboard container"
    echo "  stop      - Stop the dashboard container"
    echo "  status    - Show container status"
    echo "  logs      - Show container logs"
    echo "  clean     - Remove container and image"
    echo ""
}

# Build image
build() {
    echo -e "${YELLOW}Building Docker image for ARM64...${NC}"
    cd "$(dirname "$0")/.."
    docker build --platform linux/arm64 \
        -f docker/Dockerfile.jetson \
        -t ${IMAGE_NAME}:latest .
    echo -e "${GREEN}Build completed!${NC}"
}

# Export image
export_image() {
    echo -e "${YELLOW}Exporting image to ${TAR_FILE}...${NC}"
    docker save -o ${TAR_FILE} ${IMAGE_NAME}:latest
    echo -e "${GREEN}Export completed! Transfer ${TAR_FILE} to Jetson device.${NC}"
}

# Load image
load_image() {
    echo -e "${YELLOW}Loading image from ${TAR_FILE}...${NC}"
    sudo docker load -i ${TAR_FILE}
    echo -e "${GREEN}Image loaded successfully!${NC}"
}

# Start container
start() {
    echo -e "${YELLOW}Starting dashboard container...${NC}"

    # Stop existing container if running
    sudo docker stop ${CONTAINER_NAME} 2>/dev/null || true
    sudo docker rm ${CONTAINER_NAME} 2>/dev/null || true

    # Run new container
    sudo docker run -d \
        --name ${CONTAINER_NAME} \
        --runtime nvidia \
        -p 8050:8050 \
        --restart unless-stopped \
        ${IMAGE_NAME}:latest

    echo -e "${GREEN}Dashboard started!${NC}"
    echo -e "Access at: ${YELLOW}http://localhost:8050${NC}"
}

# Stop container
stop() {
    echo -e "${YELLOW}Stopping dashboard container...${NC}"
    sudo docker stop ${CONTAINER_NAME}
    echo -e "${GREEN}Dashboard stopped.${NC}"
}

# Show status
status() {
    echo -e "${YELLOW}Container Status:${NC}"
    sudo docker ps -a --filter "name=${CONTAINER_NAME}"
}

# Show logs
logs() {
    sudo docker logs -f ${CONTAINER_NAME}
}

# Clean up
clean() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    sudo docker stop ${CONTAINER_NAME} 2>/dev/null || true
    sudo docker rm ${CONTAINER_NAME} 2>/dev/null || true
    sudo docker rmi ${IMAGE_NAME}:latest 2>/dev/null || true
    echo -e "${GREEN}Cleanup completed.${NC}"
}

# Main
case "$1" in
    build)
        build
        ;;
    export)
        export_image
        ;;
    load)
        load_image
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    clean)
        clean
        ;;
    *)
        usage
        exit 1
        ;;
esac
