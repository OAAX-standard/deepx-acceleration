#!/bin/bash

set -e

cd "$(dirname "$0")"

# Ensure the artifacts directory exists
mkdir -p artifacts

# Read version from the version file
VERSION_FILE="../VERSION"
if [ ! -f "$VERSION_FILE" ]; then
    echo "Version file not found: $VERSION_FILE"
    exit 1
fi
VERSION=$(<"$VERSION_FILE")

# Build the toolchain as a Docker image
docker build -t "oaax-deepx-toolchain:${VERSION}" .

# Tag the versioned image as 'latest' so it can be used as the default tag
docker tag "oaax-deepx-toolchain:${VERSION}" "oaax-deepx-toolchain:latest"

# Save the Docker image as a tarball with version information
docker save "oaax-deepx-toolchain:${VERSION}" -o "./artifacts/oaax-deepx-toolchain.tar"
