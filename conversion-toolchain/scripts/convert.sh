#!/bin/bash
set -o pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <zip file> <output dir>" >&2
  exit 1
fi

# Verify dxcom is installed
if ! command -v dxcom &>/dev/null; then
  echo "Error: dxcom command not found. Ensure dx_com is installed via pip." >&2
  exit 1
fi

mkdir -p "$2"

LOG_FILE="$2/convert.log"
CURRENT_TIME=$(date "+%Y-%m-%d %H:%M:%S %Z")
echo "Starting conversion at $CURRENT_TIME" | tee -a "$LOG_FILE"

ZIP_FILE="$1"

# Extract to a temporary working directory
WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT
unzip -q "$ZIP_FILE" -d "$WORK_DIR"

onnx_file=""
json_file=""
onnx_count=0
json_count=0

for item in "$WORK_DIR"/*; do
  if [ -f "$item" ]; then
    case "${item##*.}" in
      onnx)
        onnx_file="$item"
        ((onnx_count++))
        ;;
      json)
        json_file="$item"
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
cd "$WORK_DIR"
if ! dxcom -m "$onnx_file" -c "$json_file" -o "$2" 2>&1 \
  | tee >(perl -pe 's/\r/\n/g; s/\e\[[0-9;?]*[ -\/]*[@-~]//g' >> "$LOG_FILE"); then
  echo "Error: dxcom conversion failed." | tee -a "$LOG_FILE" >&2
  exit 1
fi

CURRENT_TIME=$(date "+%Y-%m-%d %H:%M:%S %Z")
echo "Conversion finished at $CURRENT_TIME" | tee -a "$LOG_FILE"
