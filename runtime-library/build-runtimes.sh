#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPPORTED_UBUNTU_VERSIONS=("20.04" "22.04" "24.04")
CROSS_AARCH64=false
USE_CRT=false

# Error handling
error_exit() {
    echo "ERROR: $1"
    exit 1
}

# Print usage
show_usage() {
    cat << EOF
Usage: $0 --ubuntu_version <ubuntu_version> [--cross-aarch64] [--crt]
  --ubuntu_version:           (Required) Ubuntu version (${SUPPORTED_UBUNTU_VERSIONS[*]})
  --cross-aarch64: (Optional) Cross-compile for aarch64 (only supported on x86_64 hosts)
  --crt:          (Optional) Use SSL certificate (only required for DEEPX internal builds)
  -h, --help:     Show this help message

Examples:
  $0 --ubuntu_version 22.04                         # Build for host architecture
  $0 --ubuntu_version 22.04 --cross-aarch64         # Cross-compile for aarch64 (x86_64 host only)
EOF
    exit 1
}

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

# Function to read DXRT version from release.ver file
read_dxrt_version() {
    local release_file="dx_rt/release.ver"
    
    if [[ ! -f "$release_file" ]]; then
        error_exit "DX-RT release file not found: $release_file. Please ensure dx_rt directory exists and contains release.ver file."
    fi
    
    local version=$(cat "$release_file" | tr -d '[:space:]')
    if [[ -z "$version" ]]; then
        error_exit "Failed to read version from $release_file. File appears to be empty."
    fi
    
    echo "$version"
}

# Disable Python option in DX-RT cmake configuration
disable_dxrt_python_option() {
    local cmake_file="dx_rt/cmake/dxrt.cfg.cmake"
    
    if [[ ! -f "$cmake_file" ]]; then
        error_exit "DX-RT cmake configuration file not found: $cmake_file"
    fi
    
    # Create backup of original file
    cp "$cmake_file" "${cmake_file}.backup"
    
    # Replace ON with OFF for Python option
    sed -i 's/option(USE_PYTHON "Use Python" ON)/option(USE_PYTHON "Use Python" OFF)/' "$cmake_file"
    
    echo "Python option disabled in $cmake_file (backup saved as ${cmake_file}.backup)"
}

validate_files

# Read DXRT version from release.ver file
DXRT_VERSION=$(read_dxrt_version)
echo "DX-RT version read from dx_rt/release.ver: $DXRT_VERSION"

disable_dxrt_python_option

while [[ $# -gt 0 ]]; do
    case $1 in
        --ubuntu_version) UBUNTU_VERSION="$2"; shift 2 ;;
        --cross-aarch64) CROSS_AARCH64=true; shift ;;
        --crt) USE_CRT=true; shift ;;
        -h|--help) show_usage ;;
        *) error_exit "Unknown option: $1" ;;
    esac
done

# Validate required arguments
[[ -z "$UBUNTU_VERSION" ]] && error_exit "--ubuntu_version argument is required"

# Validate Ubuntu version
if [[ ! " ${SUPPORTED_UBUNTU_VERSIONS[*]} " =~ " ${UBUNTU_VERSION} " ]]; then
    error_exit "Unsupported Ubuntu version '$UBUNTU_VERSION'. Supported: ${SUPPORTED_UBUNTU_VERSIONS[*]}"
fi

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

# Determine target architecture and Dockerfile
if [[ "$CROSS_AARCH64" == true ]]; then
    TARGET_ARCH="aarch64"
    if [[ "$USE_CRT" == true ]]; then
        DOCKERFILE="Dockerfile.cross-aarch64.crt"
    else
        DOCKERFILE="Dockerfile.cross-aarch64"
    fi
else
    TARGET_ARCH="$HOST_ARCH"
    if [[ "$USE_CRT" == true ]]; then
        DOCKERFILE="Dockerfile.native.crt"
    else
        DOCKERFILE="Dockerfile.native"
    fi
fi

# Container and build setup
CONTAINER_NAME="runtime_library_${TARGET_ARCH}_ubuntu${UBUNTU_VERSION//./}"
ARTIFACTS_DIR="artifacts/${TARGET_ARCH}-ubuntu${UBUNTU_VERSION}"

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
    echo "Host Architecture: $HOST_ARCH, Target: $TARGET_ARCH, Ubuntu: $UBUNTU_VERSION, DX-RT Version: $DXRT_VERSION"
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

    # Read oaax_runtime_version from VERSION file
    if [[ ! -f "$SCRIPT_DIR/../VERSION" ]]; then
        error_exit "VERSION file not found in $SCRIPT_DIR/../VERSION"
    fi
    OAAX_RUNTIME_VERSION=$(cat "$SCRIPT_DIR/../VERSION")

    # Build Docker image using current directory as context
    echo "===== Starting Docker image build ====="
    echo "Using Dockerfile: $DOCKERFILE"
    docker build --no-cache -t "$CONTAINER_NAME" \
                 --build-arg UBUNTU_VERSION="$UBUNTU_VERSION" \
                 --network=host \
                 --progress=plain \
                 -f "$DOCKERFILE" \
                 --build-arg OAAX_RUNTIME_VERSION="$OAAX_RUNTIME_VERSION" \
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
    tar czvf library-dxrt-$DXRT_VERSION.tar.gz libRuntimeLibrary.so
    echo "Library package location: $ARTIFACTS_DIR/library-dxrt-$DXRT_VERSION.tar.gz"
}

# Main execution
main() {
    echo "========================================"
    echo "Build execution started at $(date)"
    echo "========================================"
    
    setup_artifacts_dir
    build_in_docker
    
    echo "Build completed successfully. Output in: $ARTIFACTS_DIR"
}

# Execute main function
main "$@"
