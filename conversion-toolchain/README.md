# ONNX to DXNN Conversion Toolchain

This toolchain converts ONNX models to DeepX Neural Network (DXNN) format for deployment on DeepX acceleration platforms. The conversion process optimizes models for efficient inference on DeepX hardware.

---
## Overview

The DeepX conversion toolchain consists of **three main phases**:

1. **Docker Image Build**  
   Creates a containerized environment with the DeepX compiler (DX-COM) pre-installed.

2. **Model Preparation**  
   Packages your ONNX model, JSON configuration file, and calibration dataset into a ZIP archive.

3. **Model Conversion**  
   Converts the packaged ONNX model to DXNN format using the DX-COM in the Docker container.

---
## Getting Started

### Step 1: Build the Docker Image

Build the Docker image that contains the DeepX compiler and performs the conversion from ONNX to DXNN.

> ⚠️ **Prerequisites**: 
> - Docker must be installed on your system
> - Internet access is required (the build downloads DX-COM from `sdk.deepx.ai`)

#### Example
```bash
./build-toolchain.sh
```

#### Build Process

1. **Downloads** the DX-COM Python wheel from `sdk.deepx.ai` and installs it via pip
2. **Builds** the Docker environment using the provided [Dockerfile](Dockerfile)
3. **Saves** the resulting Docker image as a `.tar` archive in the `artifacts/` directory

> 💡 The Docker image is tagged as both `oaax-deepx-toolchain:{version}` and `oaax-deepx-toolchain:latest`

#### DX-COM Version Configuration

The DX-COM version defaults to `2.3.0` and can be overridden in three ways:

```bash
# 1. Use default version
./build-toolchain.sh

# 2. Specify version as argument
./build-toolchain.sh 2.4.0

# 3. Specify version via environment variable
DX_COM_VERSION=2.4.0 ./build-toolchain.sh
```

#### Build Output

After successful execution of `build-toolchain.sh`, you'll find the Docker image archive ready for use in the `artifacts/` directory: `artifacts/oaax-deepx-toolchain.tar`

---

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
   └── calibration_dataset/ (optional)
   ```

> 💡 The .json file specifies the model's input configuration and calibration settings needed during its conversion.

#### Calibration Dataset

- **Purpose**: Used for Post-Training Quantization (PTQ) during ONNX to DXNN conversion
- **Contents**: Consists of multiple image files used as input samples for quantization
- **Configuration**: Calibration behavior is controlled by settings defined in the accompanying JSON file
- **Impact**: Not including a calibration dataset may result in degraded performance of the DXNN model

#### Using Sample Models (Optional)

Sample models and a calibration dataset are available from `sdk.deepx.ai` for testing:

```bash
# Download a sample model (ONNX + JSON config)
mkdir -p sample
curl -fL -o sample/YOLOV5S-1.onnx "https://sdk.deepx.ai/modelzoo/onnx/YOLOV5S-1.onnx"
curl -fL -o sample/YOLOV5S-1.json "https://sdk.deepx.ai/modelzoo/json/YOLOV5S-1.json"

# Download the calibration dataset
curl -fL -o calibration_dataset.tar.gz "https://sdk.deepx.ai/dataset/calibration_dataset.tar.gz"
mkdir -p calibration_dataset
tar xfz calibration_dataset.tar.gz --strip-components=1 -C calibration_dataset/
rm calibration_dataset.tar.gz
```

> ⚠️ **Sample JSON update required**: The downloaded sample JSON currently uses an environment-specific `dataset_path` such as `"/mnt/datasets/COCO"`. If you use the downloaded `calibration_dataset/` folder from the example above, update the JSON file so that `dataset_path` is `"./calibration_dataset"` before creating the ZIP archive.

Example:
```json
"dataset_path": "./calibration_dataset"
```

#### Example: Create a ZIP archive from sample files

```bash
mkdir -p artifacts
rm -f artifacts/YOLOV5S-1.zip
(cd sample && zip -j ../artifacts/YOLOV5S-1.zip YOLOV5S-1.onnx YOLOV5S-1.json)
zip -r artifacts/YOLOV5S-1.zip calibration_dataset
```

> ⚠️ **Note**: The first command uses `-j` so the ONNX and JSON files are stored at the ZIP root. The second command adds the `calibration_dataset/` directory with its folder name preserved.

---

### Step 3: Run Model Conversion

Once the Docker image is built and your model is prepared as a `.zip` archive, you can convert your ONNX model to DXNN format.

#### Example
```bash
docker run -v ./artifacts:/app/artifacts oaax-deepx-toolchain:latest /app/artifacts/YOLOV5S-1.zip /app/artifacts
```
> ⚠️ **Note**: Run this command from the `conversion-toolchain/` directory so that `./artifacts` resolves correctly on the host. The file `YOLOV5S-1.zip` must exist under that `./artifacts` directory. Refer to the example command in Step 2 for how to generate this file.

######  Command Breakdown
- `-v ./artifacts:/app/artifacts`: Mounts the `artifacts/` directory from the host machine to `/app/artifacts` inside the Docker container
- `oaax-deepx-toolchain:latest`: The Docker image to use
- `/app/artifacts/YOLOV5S-1.zip`: Path to the `YOLOV5S-1.zip` file inside the Docker container
- `/app/artifacts`: Output directory inside the Docker container

#### Conversion Process

1. **File Extraction**: The ZIP file is extracted to access ONNX and JSON files
2. **Validation**: Checks for exactly one ONNX file and one JSON file
3. **Compilation**: Uses DX-COM (`dxcom` CLI) to convert ONNX to DXNN format
4. **Output Generation**: Creates the converted model and log files

#### Conversion Output

After successful conversion, you'll find in the `artifacts/` directory:
- `*.dxnn`: The converted DeepX Neural Network (DXNN) model
- `convert.log`: Detailed conversion log with timestamps and process information
