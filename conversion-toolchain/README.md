# ONNX to DXNN Conversion Toolchain

This toolchain converts ONNX models to DeepX Neural Network (DXNN) format for deployment on DeepX acceleration platforms. The conversion process optimizes models for efficient inference on DeepX hardware.

---
## Overview

The DeepX conversion toolchain consists of **three main phases**:

1. **Docker Image Build**  
   Creates a containerized environment with the DeepX compiler (DX-COM).

2. **Model Preparation**  
   Packages your ONNX model, JSON configuration file, and calibration dataset into a ZIP archive.

3. **Model Conversion**  
   Converts the packaged ONNX model to DXNN format using the DX-COM inside the Docker container.

---
## Getting Started

### Step 1: Download DEEPX's DX-COM

The first step is to download the DeepX compiler (DX-COM) package using the provided script and unpack it into the `dx_com/` directory. This package is required for converting ONNX models to DXNN format.

> ⚠️ **Prerequisites**: 
> - Python 3 with `requests` and `beautifulsoup4` packages installed (`pip install requests beautifulsoup4`) (used by [downloader.py](../scripts/downloader.py))
> - Valid credentials for the [DeepX Developer Portal](https://developer.deepx.ai/) (required to download DX-COM)

#### Examples

##### **Using default settings:**
```bash
DX_USERNAME="your_username" DX_PASSWORD="your_password" ./setup-dx_com.sh
```
> ⚠️ **Authentication Required**: You need a [DeepX Developer Portal](https://developer.deepx.ai/) account to download DX-COM. 
> - The script will prompt for credentials if not provided via environment variables
> - For CI/CD workflows, use environment variables `DX_USERNAME` and `DX_PASSWORD`

##### **Specifying expected version and download URL:**
```bash
./setup-dx_com.sh --expected-version "1.60.1" --download-url "https://developer.deepx.ai/?files=MjM2NA=="
```

###### Command Line Options

- `--expected-version <version>`: Specify the DX-COM version string expected in the downloaded filename (default: 1.60.1).
This value must match the version string in the downloaded archive filename.
**If the version does not match, the download will be considered failed**.
- `--download-url <url>`: Specify download URL for DX-COM (default: https://developer.deepx.ai/?files=MjM2NA==)
- `--help`: Show help message with all available options

#### Setup Output

After successful execution of `setup-dx_com.sh`, you'll find the following:

- `dx_com/` - DeepX compiler files extracted from the downloaded package
- `sample/` - Example ONNX models and configuration files for testing
- `calibration_dataset/` - Sample calibration dataset for quantization

### Step 2: Build the Docker Image

The second step is to build the Docker image that performs the conversion from ONNX to DXNN.

> ⚠️ **Prerequisites**: 
> - Docker must be installed on your system

#### Build Process

1. **Copies** the [convert.sh](scripts/convert.sh) entrypoint script into the Docker image  
3. **Builds** the Docker environment using the provided [Dockerfile](Dockerfile)
4. **Saves** the resulting Docker image as a `.tar` archive in the `artifacts/` directory

> 💡 The Docker image is tagged as both `oaax-deepx-toolchain:{version}` and `oaax-deepx-toolchain:latest`

#### Build Output

After successful execution of `build-toolchain.sh`, you'll find the Docker image archive ready for use in the `artifacts/` directory: `artifacts/oaax-deepx-toolchain.tar`

### Step 2: Prepare Your Model

Before conversion, package your ONNX model into a `.zip` archive following the required format:

#### Model Packaging Requirements

1. **ZIP Archive**: Must include the following files and directories directly at the root of the archive:
   - Exactly one ONNX file (`*.onnx`)
   - Exactly one JSON configuration file (`*.json`)
   - Optional but recommended: Calibration dataset folder

2. **Correct ZIP Structure**:
   ```
   your_model.zip
   ├── your_model.onnx
   ├── your_model.json
   └── calibration dataset/ (optional)
   ```

#### Sample Models

After Step 1, a `sample/` directory will be created automatically when DX-COM is extracted. It contains example models such as:
- `YOLOV5-1.onnx` + `YOLOV5-1.json`
- `ResNet50-1.onnx` + `ResNet50-1.json`
- `MobileNetV1-1.onnx` + `MobileNetV1-1.json`

> 💡 The .json file specifies the model’s input configuration and calibration settings needed during its conversion.

#### Calibration Dataset

In addition to the `sample/` directory, a `calibration_dataset/` directory is also generated automatically when DX-COM is extracted.

- **Purpose**: Used for Post-Training Quantization (PTQ) during ONNX to DXNN conversion
- **Contents**: Consists of multiple image files used as input samples for quantization
- **Configuration**: Calibration behavior is controlled by settings defined in the accompanying JSON file
- **Impact**: Not including a calibration dataset may result in degraded performance of the DXNN model

#### Example
The following steps use a sample YOLOv5 model along with its calibration dataset to create a zip archive named `YOLOV5-1.zip` under the `artifacts/` directory.
   ```
   mkdir -p temp_zip
   cp sample/YOLOV5-1.onnx temp_zip/
   cp sample/YOLOV5-1.json temp_zip/
   cp -r calibration_dataset temp_zip/

   cd temp_zip
   zip -r ../artifacts/YOLOV5-1.zip *
   cd ..
   rm -rf temp_zip
   ```
> ⚠️ **Note**: Be sure to use `*` instead of `.` in the `zip` command.
Using `.` will include the current directory itself in the archive, resulting in an incorrect structure, which may break the converter.



### Step 3: Run Model Conversion

Once the Docker image is built and your model is prepared, you can convert your ONNX model to DXNN format.

#### Example
```bash
docker run -v ./artifacts:/app/artifacts -v ./dx_com:/app/dx_com oaax-deepx-toolchain:latest /app/artifacts/YOLOV5-1.zip /app/artifacts
```
> ⚠️ **Note**: The file `YOLOV5-1.zip` must exist under the `./artifacts` directory.
Refer to the example command in Step 2 for how to generate this file.

######  Command Breakdown
- `-v ./artifacts:/app/artifacts`: Mounts the `artifacts/` directory from the host machine to `/app/artifacts` inside the Docker container
- `-v ./dx_com:/app/dx_com`: Mounts the `dx_com/` directory from the host machine to `/app/dx_com` inside the Docker container
- `oaax-deepx-toolchain:latest`: The Docker image to use
- `/app/artifacts/YOLOV5-1.zip`: Path to the `YOLOV5-1.zip` file inside the Docker container
- `/app/artifacts`: Output directory inside the Docker container

#### Conversion Process

1. **File Extraction**: The ZIP file is extracted to access ONNX and JSON files
2. **Validation**: Checks for exactly one ONNX file and one JSON file
3. **Compilation**: Uses `dx_com` to convert ONNX to DXNN format
4. **Output Generation**: Creates the converted model and log files

#### Conversion Output

After successful conversion, you'll find in the `artifacts/` directory:
- `*.dxnn`: The converted DeepX Neural Network (DXNN) model
- `convert.log`: Detailed conversion log with timestamps and process information

