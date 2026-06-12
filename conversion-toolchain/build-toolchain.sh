#!/bin/bash

set -e

cd "$(dirname "$0")"

# DX-COM version (override via argument or environment variable)
# Usage:
#   ./build-toolchain.sh                      # uses default version
#   ./build-toolchain.sh 2.4.0                # specify version as argument
#   DX_COM_VERSION=2.4.0 ./build-toolchain.sh # specify version via env var
DX_COM_VERSION="${1:-${DX_COM_VERSION:-2.3.0}}"

# Ensure the artifacts directory exists
mkdir -p artifacts

# Read version from the version file
VERSION_FILE="../VERSION"
if [ ! -f "$VERSION_FILE" ]; then
    echo "Version file not found: $VERSION_FILE"
    exit 1
fi
VERSION=$(<"$VERSION_FILE")

echo "Building oaax-deepx-toolchain:${VERSION} with DX-COM v${DX_COM_VERSION}"

# Build the toolchain as a Docker image
docker build \
    --build-arg DX_COM_VERSION="${DX_COM_VERSION}" \
    -t "oaax-deepx-toolchain:${VERSION}" .

# Tag the versioned image as 'latest' so it can be used as the default tag
docker tag "oaax-deepx-toolchain:${VERSION}" "oaax-deepx-toolchain:latest"

# Save both tags so loading the tarball restores the versioned and latest image names
docker save \
    "oaax-deepx-toolchain:${VERSION}" \
    "oaax-deepx-toolchain:latest" \
    -o "./artifacts/oaax-deepx-toolchain.tar"
