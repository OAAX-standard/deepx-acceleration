#!/bin/bash

set -e

# DeepX SDK configuration (default values)
DX_COM_VERSION="2.0.0"
DX_COM_DOWNLOAD_URL="https://developer.deepx.ai/?files=MjQ1Mw=="

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --expected-version)
            DX_COM_VERSION="$2"
            shift 2
            ;;
        --download-url)
            DX_COM_DOWNLOAD_URL="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--expected-version <version>] [--download-url <url>]"
            echo ""
            echo "Options:"
            echo "  --expected-version <version>       Expected DX-COM version to download (default: 2.0.0)"
            echo "  --download-url <url>      Download URL for DX-COM (default: https://developer.deepx.ai/?files=MjQ1Mw==)"
            echo "  --help                    Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  DX_USERNAME              DeepX Developer Portal (developer.deepx.ai) username"
            echo "  DX_PASSWORD              DeepX Developer Portal (developer.deepx.ai) password"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "DeepX Compiler (DX-COM) Download Tool"
echo "======================="
echo "Version: $DX_COM_VERSION"
echo "Download URL: $DX_COM_DOWNLOAD_URL"
echo ""

# Get credentials from environment variables or prompt user
if [[ -n "$DX_USERNAME" && -n "$DX_PASSWORD" ]]; then
    echo "Using credentials from environment variables"
elif [[ -n "$DX_USERNAME" ]]; then
    echo "Username found in environment variable, prompting for password"
    read -s -p "(developer.deepx.ai) Enter your password: " DX_PASSWORD
    echo
elif [[ -n "$DX_PASSWORD" ]]; then
    echo "Password found in environment variable, prompting for username"
    read -p "(developer.deepx.ai) Enter your username: " DX_USERNAME
else
    echo "No credentials found in environment variables, prompting for both"
    read -p "(developer.deepx.ai) Enter your username: " DX_USERNAME
    read -s -p "(developer.deepx.ai) Enter your password: " DX_PASSWORD
    echo
fi

# Validate credentials
if [[ -z "$DX_USERNAME" || -z "$DX_PASSWORD" ]]; then
    echo "Error: Username and password are required"
    echo "You can set them as environment variables: DX_USERNAME and DX_PASSWORD"
    exit 1
fi

cd "$(dirname "$0")" || exit 1

rm -rf artifacts 2&> /dev/null || true
rm -rf calibration_dataset 2&> /dev/null || true
rm -rf dx_com 2&> /dev/null || true
rm -rf sample 2&> /dev/null || true

mkdir artifacts

# Download and extract DeepX SDK using downloader.py
echo "Downloading DX-COM v$DX_COM_VERSION using authenticated download..."

# Check if required Python packages are available
if ! python3 -c "import requests, bs4" 2>/dev/null; then
    echo "Error: Required Python packages (requests, bs4) are not installed"
    echo "Please install them using: pip install requests beautifulsoup4"
    exit 1
fi

# Create temporary download directory
TEMP_DOWNLOAD_DIR="temp_download"
rm -rf "$TEMP_DOWNLOAD_DIR" 2>/dev/null || true
mkdir -p "$TEMP_DOWNLOAD_DIR"

# Ensure that downloader.py exists in scripts directory
DOWNLOADER_SCRIPT="../scripts/downloader.py"
if [[ ! -f "$DOWNLOADER_SCRIPT" ]]; then
    echo "Error: downloader.py script not found in the 'scripts' directory in the parent directory"
    exit 1
fi

# Download using downloader.py
echo "Authenticating and downloading from DeepX developer portal..."
if python3 "$DOWNLOADER_SCRIPT" \
    --username "$DX_USERNAME" \
    --password "$DX_PASSWORD" \
    --download-url "$DX_COM_DOWNLOAD_URL" \
    --save-location "$TEMP_DOWNLOAD_DIR" \
    --expected-version "$DX_COM_VERSION"; then
    
    echo "Download completed successfully"
    
    # Find the downloaded file
    DOWNLOADED_FILE=$(find "$TEMP_DOWNLOAD_DIR" -name "*.tar.gz" | head -1)
    if [[ -z "$DOWNLOADED_FILE" ]]; then
        echo "Error: Could not find downloaded tar.gz file"
        exit 1
    fi
    
    echo "Found downloaded file: $DOWNLOADED_FILE"
    
    # Create extraction directory
    EXTRACT_DIR="dx_com_M1_v${DX_COM_VERSION}"
    echo "Creating extraction directory: $EXTRACT_DIR"
    rm -rf "$EXTRACT_DIR" 2>/dev/null || true
    mkdir "$EXTRACT_DIR"
    
    echo "Extracting SDK file into $EXTRACT_DIR..."
    if tar -xzf "$DOWNLOADED_FILE" -C "$EXTRACT_DIR"; then
        echo "Extraction completed successfully"
        
        # Organize extracted files - move required directories to root
        echo "Organizing extracted files..."
        
        # Remove existing directories if they exist
        rm -rf dx_com sample calibration_dataset 2>/dev/null || true
        
        # Find and move dx_com directory
        DX_COM_PATH=$(find "$EXTRACT_DIR" -type d -name "dx_com" | head -1)
        if [ -n "$DX_COM_PATH" ]; then
            echo "Found dx_com directory at: $DX_COM_PATH"
            mv "$DX_COM_PATH" .
        else
            echo "Error: Could not find dx_com directory in extracted files"
            exit 1
        fi
        
        # Find and move sample directory
        SAMPLE_PATH=$(find "$EXTRACT_DIR" -type d -name "sample" | head -1)
        if [ -n "$SAMPLE_PATH" ]; then
            echo "Found sample directory at: $SAMPLE_PATH"
            mv "$SAMPLE_PATH" .
        else
            echo "Warning: Could not find sample directory in extracted files"
        fi
        
        # Find and move calibration_dataset directory
        CALIB_PATH=$(find "$EXTRACT_DIR" -type d -name "calibration_dataset" | head -1)
        if [ -n "$CALIB_PATH" ]; then
            echo "Found calibration_dataset directory at: $CALIB_PATH"
            mv "$CALIB_PATH" .
        else
            echo "Warning: Could not find calibration_dataset directory in extracted files"
        fi
        
        # Clean up extraction directory and temporary files
        echo "Cleaning up temporary files..."
        rm -rf "$EXTRACT_DIR"
        rm -rf "$TEMP_DOWNLOAD_DIR"
        
        echo "File organization completed successfully"
        
    else
        echo "Error: Failed to extract SDK file"
        exit 1
    fi
else
    echo "Error: Failed to download SDK using downloader.py"
    rm -rf "$TEMP_DOWNLOAD_DIR" 2>/dev/null || true
    exit 1
fi