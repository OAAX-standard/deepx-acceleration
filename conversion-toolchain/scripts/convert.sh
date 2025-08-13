#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <zip file> <output dir>" >&2
  exit 1
fi

# Ensure the dx_com directory exists
if [ ! -d "./dx_com" ]; then
  echo "Error: dx_com directory not found. Please download it, unpack it, then mount it into the Docker container at /app/dx_com." >&2
  exit 1
fi

mkdir -p "$2"

LOG_FILE="$2/convert.log"
CURRENT_TIME=$(date "+%Y-%m-%d %H:%M:%S %Z")
echo "Starting conversion at $CURRENT_TIME" | tee -a "$LOG_FILE"

ZIP_FILE="$1"

unzip "$ZIP_FILE"

onnx_file=""
json_file=""
onnx_count=0
json_count=0

for item in *; do
  if [ -f "$item" ]; then
    case "${item##*.}" in
      onnx)
        onnx_file=$(realpath "$item")
        ((onnx_count++))
        ;;
      json)
        json_file=$(realpath "$item")
        ((json_count++))
        ;;
    esac
  fi
done

if [ "$onnx_count" -ne 1 ]; then
  echo "Error: Expected exactly 1 ONNX file, but found $onnx_count." >&2
  exit 1
fi

if [ "$json_count" -ne 1 ]; then
  echo "Error: Expected exactly 1 JSON file, but found $json_count." >&2
  exit 1
fi

echo "ONNX file: $onnx_file" | tee -a "$LOG_FILE"
echo "JSON file: $json_file" | tee -a "$LOG_FILE"

echo "Converting $onnx_file to DXNN" | tee -a "$LOG_FILE"
./dx_com/dx_com -m "$onnx_file" -c "$json_file" -o "$2" | tee -a "$LOG_FILE"

CURRENT_TIME=$(date "+%Y-%m-%d %H:%M:%S %Z")
echo "Conversion finished at $CURRENT_TIME" | tee -a "$LOG_FILE"
