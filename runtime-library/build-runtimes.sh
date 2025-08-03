#!/bin/bash

# Configuration
SUPPORTED_UBUNTU_VERSIONS=("20.04" "22.04" "24.04")

# Error handling
error_exit() {
    echo "ERROR: $1"
    exit 1
}

# Print usage
show_usage() {
    cat << EOF
Usage: $0 --os <ubuntu_version> [--cross-aarch64] [--dxrt-version <version>] [--crt]
  --os:           (Required) Ubuntu version (${SUPPORTED_UBUNTU_VERSIONS[*]})
  --cross-aarch64: (Optional) Cross-compile for aarch64 (only supported on x86_64 hosts)
  --dxrt-version: (Optional) Version of DX RT used to build the runtime. Default is 2.9.5.
  --crt:          (Optional) Use SSL certificate (only required for DEEPX internal builds)
  -h, --help:     Show this help message

Examples:
  $0 --os 22.04                         # Build for host architecture
  $0 --os 22.04 --cross-aarch64         # Cross-compile for aarch64 (x86_64 host only)
  $0 --os 22.04 --dxrt-version 2.9.5    # Cross-compile for aarch64 (x86_64 host only)
EOF
    exit 1
}

# Command line argument parsing
CROSS_AARCH64=false
USE_CRT=false
DXRT_VERSION=2.9.5

while [[ $# -gt 0 ]]; do
    case $1 in
        --os) UBUNTU_VERSION="$2"; shift 2 ;;
        --cross-aarch64) CROSS_AARCH64=true; shift ;;
        --dxrt-version) DXRT_VERSION="$2"; shift 2 ;;
        --crt) USE_CRT=true; shift ;;
        -h|--help) show_usage ;;
        *) error_exit "Unknown option: $1" ;;
    esac
done

# Validate required arguments
[[ -z "$UBUNTU_VERSION" ]] && error_exit "--os argument is required"

# Validate Ubuntu version
if [[ ! " ${SUPPORTED_UBUNTU_VERSIONS[*]} " =~ " ${UBUNTU_VERSION} " ]]; then
    error_exit "Unsupported Ubuntu version '$UBUNTU_VERSION'. Supported: ${SUPPORTED_UBUNTU_VERSIONS[*]}"
fi

# Setup variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Architecture detection
HOST_ARCH=$(uname -m)
case "$HOST_ARCH" in
    x86_64) ;;
    aarch64) 
        if [[ "$CROSS_AARCH64" == true ]]; then
            error_exit "Cross-compilation for aarch64 is not supported on aarch64 hosts"
        fi
        ;;
    *) error_exit "Unsupported host architecture: $HOST_ARCH" ;;
esac

# Print build configuration
echo "Build configuration:"
echo "  Host Architecture: $HOST_ARCH"
echo "  CROSS_AARCH64: $CROSS_AARCH64"
echo "  Ubuntu Version: $UBUNTU_VERSION"
echo "  DX RT Version: $DXRT_VERSION"

# Determine target architecture and Dockerfile
if [[ "$CROSS_AARCH64" == true ]]; then
    TARGET_ARCH="aarch64"
    if [[ "$USE_CRT" == true ]]; then
        DOCKERFILE="Dockerfile.cross-aarch64.crt"
    else
        DOCKERFILE="Dockerfile.cross-aarch64"
    fi
    echo "Cross-compiling for aarch64 on x86_64 host"
else
    TARGET_ARCH="$HOST_ARCH"
    if [[ "$USE_CRT" == true ]]; then
        DOCKERFILE="Dockerfile.native.crt"
    else
        DOCKERFILE="Dockerfile.native"
    fi
    echo "Building for host architecture: $HOST_ARCH"
fi

# Container and build setup
CONTAINER_NAME="runtime_library_${TARGET_ARCH}_ubuntu${UBUNTU_VERSION//./}"
ARTIFACTS_DIR="artifacts/${TARGET_ARCH}-ubuntu${UBUNTU_VERSION}"

# File validation function
validate_files() {
    local files_to_check=(
        "$SCRIPT_DIR/dx_rt"
    )
    
    # Check for certificate file if --crt option is used
    if [[ "$USE_CRT" == true ]]; then
        if [[ ! -f "$SCRIPT_DIR/Fortinet_CA_SSL.crt" ]]; then
            error_exit "Certificate file Fortinet_CA_SSL.crt not found in current directory. Required when using --crt option."
        fi
        echo "Certificate file found: $SCRIPT_DIR/Fortinet_CA_SSL.crt"
        files_to_check+=("$SCRIPT_DIR/Fortinet_CA_SSL.crt")
    fi
    
    for file in "${files_to_check[@]}"; do
        [[ ! -e "$file" ]] && error_exit "Required file/directory not found: $file"
    done
}

# Setup artifacts directory
setup_artifacts_dir() {
    echo "Setting up artifacts directory: $ARTIFACTS_DIR"
    
    # Create artifacts base directory if it doesn't exist
    mkdir -p "$SCRIPT_DIR/artifacts" || error_exit "Failed to create artifacts base directory"
    
    # Remove only the specific target directory if it exists
    if [[ -d "$SCRIPT_DIR/$ARTIFACTS_DIR" ]]; then
        rm -rf "$SCRIPT_DIR/$ARTIFACTS_DIR" || error_exit "Failed to remove existing target directory"
    fi
    
    mkdir -p "$SCRIPT_DIR/$ARTIFACTS_DIR" || error_exit "Failed to create target directory"
}

# Main build function
build_in_docker() {
    echo "===== Build process started ====="
    echo "Host Architecture: $HOST_ARCH, Target: $TARGET_ARCH, Ubuntu: $UBUNTU_VERSION"
    if [[ "$USE_CRT" == true ]]; then
        echo "SSL certificate installation: Enabled"
    else
        echo "SSL certificate installation: Disabled"
    fi

    # Remove existing Docker image
    if docker image inspect "$CONTAINER_NAME" &>/dev/null; then
        echo "Removing existing Docker image: $CONTAINER_NAME"
        docker rmi -f "$CONTAINER_NAME"
    fi

    # Build Docker image using current directory as context
    echo "===== Starting Docker image build ====="
    echo "Using Dockerfile: $DOCKERFILE"
    docker build --no-cache -t "$CONTAINER_NAME" \
                 --build-arg UBUNTU_VERSION="$UBUNTU_VERSION" \
                 --network=host \
                 --progress=plain \
                 -f "$DOCKERFILE" \
                 . || error_exit "Docker image build failed"

    # Extract libRuntimeLibrary.so
    echo "===== Extracting libRuntimeLibrary.so ====="
    docker run --rm -v "$SCRIPT_DIR/$ARTIFACTS_DIR:/artifacts" \
               "$CONTAINER_NAME" \
               bash -c "cp /usr/local/lib/libRuntimeLibrary.so /artifacts/" || {
        error_exit "Failed to extract libRuntimeLibrary.so"
    }
    
    echo "===== Build process completed successfully ====="
    echo "Docker image: $CONTAINER_NAME"
    echo "Artifacts location: $ARTIFACTS_DIR/libRuntimeLibrary.so"

    echo "===== Compressing the runtime library into tar.gz archive ====="
    cd $ARTIFACTS_DIR  
    tar czvf library-$DXRT_VERSION.tar.gz libRuntimeLibrary.so
    echo "Library package location: $ARTIFACTS_DIR/library-$DXRT_VERSION.tar.gz"
}

# Main execution
main() {
    echo "========================================"
    echo "Build execution started at $(date)"
    echo "========================================"
    
    validate_files
    setup_artifacts_dir
    build_in_docker
    
    echo "Build completed successfully. Output in: $ARTIFACTS_DIR"
}

# Execute main function
main "$@"
