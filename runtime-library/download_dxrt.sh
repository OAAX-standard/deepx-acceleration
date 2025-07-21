#!/bin/bash

set -e

# DeepX Runtime configuration (default values)
DX_RT_VERSION="2.9.5"
DX_RT_DOWNLOAD_URL="https://developer.deepx.ai/?files=MjM1Ng=="

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --expected-version)
            DX_RT_VERSION="$2"
            shift 2
            ;;
        --download-url)
            DX_RT_DOWNLOAD_URL="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--expected-version <version>] [--download-url <url>]"
            echo ""
            echo "Options:"
            echo "  --expected-version <version>       Expected DX-RT version to download (default: 2.9.5)"
            echo "  --download-url <url>      Download URL for DX-RT (default: https://developer.deepx.ai/?files=MjM1Ng==)"
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

echo "DeepX Runtime (DX-RT) Download Tool"
echo "==================================="
echo "Version: $DX_RT_VERSION"
echo "Download URL: $DX_RT_DOWNLOAD_URL"
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

# Clean up existing files
rm -rf dx_rt-main.zip 2>/dev/null || true
rm -rf dx_rt-main 2>/dev/null || true
rm -rf dx_rt 2>/dev/null || true

# Download and extract DeepX Runtime using downloader.py
echo "Downloading DX-RT v$DX_RT_VERSION using authenticated download..."

# Check if downloader.py exists in scripts directory
DOWNLOADER_SCRIPT="../scripts/downloader.py"
if [[ ! -f "$DOWNLOADER_SCRIPT" ]]; then
    echo "Error: downloader.py not found at $DOWNLOADER_SCRIPT"
    exit 1
fi

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

# Download using downloader.py
echo "Authenticating and downloading from DeepX developer portal..."
if python3 "$DOWNLOADER_SCRIPT" \
    --username "$DX_USERNAME" \
    --password "$DX_PASSWORD" \
    --download-url "$DX_RT_DOWNLOAD_URL" \
    --save-location "$TEMP_DOWNLOAD_DIR" \
    --expected-version "$DX_RT_VERSION"; then
    
    echo "Download completed successfully"
    
    # Find the downloaded file
    DOWNLOADED_FILE=$(find "$TEMP_DOWNLOAD_DIR" -name "*.tar.gz" | head -1)
    if [[ -z "$DOWNLOADED_FILE" ]]; then
        echo "Error: Could not find downloaded tar.gz file"
        exit 1
    fi
    
    echo "Found downloaded file: $DOWNLOADED_FILE"
    
    # Create extraction directory
    EXTRACT_DIR="dx_rt_v${DX_RT_VERSION}_extract"
    echo "Creating extraction directory: $EXTRACT_DIR"
    rm -rf "$EXTRACT_DIR" 2>/dev/null || true
    mkdir "$EXTRACT_DIR"
    
    echo "Extracting DX-RT file into $EXTRACT_DIR..."
    if tar -xzf "$DOWNLOADED_FILE" -C "$EXTRACT_DIR"; then
        echo "Extraction completed successfully"
        
        # Organize extracted files - find and move dx_rt directory
        echo "Organizing extracted files..."
        
        # Find dx_rt directory in extracted files
        DX_RT_PATH=$(find "$EXTRACT_DIR" -type d -name "dx_rt" | head -1)
        if [ -n "$DX_RT_PATH" ]; then
            echo "Found dx_rt directory at: $DX_RT_PATH"
            mv "$DX_RT_PATH" .
        else
            echo "Error: Could not find dx_rt directory in extracted files"
            echo "Available directories:"
            find "$EXTRACT_DIR" -type d -maxdepth 2
            exit 1
        fi
        
        # Clean up extraction directory and temporary files
        echo "Cleaning up temporary files..."
        rm -rf "$EXTRACT_DIR"
        rm -rf "$TEMP_DOWNLOAD_DIR"
        
        echo "DX-RT installation completed successfully!"
        echo "Installed component:"
        echo "  - dx_rt/"
        
    else
        echo "Error: Failed to extract DX-RT file"
        exit 1
    fi
else
    echo "Error: Failed to download DX-RT using downloader.py"
    rm -rf "$TEMP_DOWNLOAD_DIR" 2>/dev/null || true
    exit 1
fi 