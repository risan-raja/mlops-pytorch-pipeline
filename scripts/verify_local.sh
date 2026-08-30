#!/usr/bin/env bash

# Drafted with the help of an AI Agent
set -euo pipefail

echo "=========================================================="
echo "   MLOps PyTorch Pipeline: Local Docker Verification      "
echo "=========================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

TEST_IMAGE="${ROOT_DIR}/test_image.png"
if [ ! -f "${TEST_IMAGE}" ]; then
    if [ -f "${ROOT_DIR}/samples/airplane.png" ]; then
        cp "${ROOT_DIR}/samples/airplane.png" "${TEST_IMAGE}"
    else
        echo "Creating placeholder test_image.png..."
        uv run python -c '
from PIL import Image
Image.new("RGB", (32, 32), color=(73, 109, 137)).save("test_image.png")
'
    fi
fi

echo ""
echo "[Step 1/5] Building training Docker image..."
docker build -f docker/Dockerfile.train -t mlops-train:v1 .

echo ""
echo "[Step 2/5] Running containerized training with mounted volumes..."
docker run --rm \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    mlops-train:v1

echo ""
echo "[Step 3/5] Building serving Docker image..."
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

echo ""
echo "[Step 4/5] Running serving container..."
CONTAINER_NAME="mlops-serve-verify-$$"
docker run -d --rm \
    --name "${CONTAINER_NAME}" \
    -p 8080:8080 \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    mlops-serve:v1

cleanup() {
    echo ""
    echo "Stopping serving container (${CONTAINER_NAME})..."
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for serving endpoint to become healthy..."
MAX_RETRIES=20
COUNT=0
until curl -s -f http://localhost:8080/health > /dev/null 2>&1; do
    sleep 1
    COUNT=$((COUNT + 1))
    if [ "${COUNT}" -ge "${MAX_RETRIES}" ]; then
        echo "Error: Serving endpoint timed out."
        docker logs "${CONTAINER_NAME}"
        exit 1
    fi
done

echo "Health check succeeded:"
curl -s http://localhost:8080/health
echo ""

echo ""
echo "[Step 5/5] Testing prediction endpoint with ${TEST_IMAGE}..."
PREDICTION_RESPONSE=$(curl -s -X POST http://localhost:8080/predict -F "image=@${TEST_IMAGE}")
echo "Prediction Response:"
echo "${PREDICTION_RESPONSE}"

echo ""
echo "=========================================================="
echo "  Local Docker Verification Completed Successfully!   "
echo "=========================================================="
