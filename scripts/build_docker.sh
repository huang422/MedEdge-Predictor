#!/bin/bash
# =============================================================================
# Docker Build Script for Medical Prediction Dashboard
# =============================================================================

set -e

# Configuration
IMAGE_NAME="medical-prediction-dashboard"
VERSION="1.0.0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Medical Prediction Dashboard - Docker Build${NC}"
echo -e "${GREEN}========================================${NC}"

# Detect platform
PLATFORM=$(uname -m)
echo -e "${YELLOW}Detected platform: ${PLATFORM}${NC}"

# Navigate to project root
cd "$(dirname "$0")/.."

# Build based on platform
if [[ "$PLATFORM" == "aarch64" ]] || [[ "$1" == "--jetson" ]]; then
    echo -e "${YELLOW}Building for NVIDIA Jetson (ARM64)...${NC}"
    docker build \
        --platform linux/arm64 \
        -f docker/Dockerfile.jetson \
        -t ${IMAGE_NAME}:${VERSION} \
        -t ${IMAGE_NAME}:latest \
        .
else
    echo -e "${YELLOW}Building for standard platform...${NC}"
    docker build \
        -f docker/Dockerfile \
        -t ${IMAGE_NAME}:${VERSION} \
        -t ${IMAGE_NAME}:latest \
        .
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "To run the dashboard:"
echo -e "  ${YELLOW}docker run -p 8050:8050 ${IMAGE_NAME}:latest${NC}"
echo ""
echo -e "Or use docker-compose:"
echo -e "  ${YELLOW}docker-compose -f docker/docker-compose.yml up${NC}"
