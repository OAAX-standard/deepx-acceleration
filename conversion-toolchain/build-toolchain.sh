#!/bin/bash

set -e

cd "$(dirname "$0")"

# Build the toolchain as a Docker image
docker build -t "onnx-to-dxnn:dx_com_v${DX_COM_VERSION}" .

# Tag the versioned image as 'latest' so it can be used as the default tag
docker tag "onnx-to-dxnn:dx_com_v${DX_COM_VERSION}" "onnx-to-dxnn:latest"

# Save the Docker image as a tarball with version information
docker save "onnx-to-dxnn:dx_com_v${DX_COM_VERSION}" -o "./artifacts/onnx-to-dxnn-dx_com_v${DX_COM_VERSION}.tar"
