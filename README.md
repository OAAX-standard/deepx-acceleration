# DEEPX OAAX Implementation


<img src="https://cdn.prod.website-files.com/6649422e9fd6f89e2b293704/6649422e9fd6f89e2b293710_logo.svg" alt="OAAX" width="49%" />
<img src="./resources/image.png" alt="DeepX Acceleration" width="49%" />

This repository contains the source code for building the implementation of the **conversion toolchain and runtime library for DeepX NPU acceleration**, based on the **OAAX standard**. 
> If you're new to OAAX, you can find more information about the OAAX standard in the [OAAX repository](https://github.com/oaax-standard/OAAX)

This project builds on top of the [reference implementation](https://github.com/OAAX-standard/reference-implementation) provided in the OAAX repository.
Please check it out first to gain a high-level understanding of how an OAAX-compliant runtime and conversion toolchain should be implemented.

---

## Getting Started

This project uses Git submodules for the [tools](https://github.com/OAAX-standard/tools) dependency. When cloning the repository, it is recommended to use the `--recursive` option to automatically initialize and update the submodules:

```bash
git clone --recursive https://github.com/OAAX-standard/deepx-acceleration.git
```

If you have already cloned the repository without the `--recursive` option, you can initialize the submodules manually:

```bash
git submodule update --init --recursive
```

---

## Repository structure

The repository is organized into two main components:

- **[`conversion-toolchain`](conversion-toolchain/)**:
Contains the scripts and Docker setup for converting ONNX models into the DXNN (DeepX Neural Network) format using the DeepX compiler (`DX-COM`).

- **[`runtime-library`](runtime-library/)**:
Provides the source code and a Docker-based build system to generate the C++ runtime library (`libRuntimeLibrary.so`), which is responsible for loading and executing DXNN models on DeepX NPUs using the DeepX Runtime (`DX-RT`).

---

## Building the implementation

You can build both the **conversion toolchain** and the **runtime library** independently using the provided shell scripts in each folder.  
Upon successful build, an `artifacts/` directory will be created in each component, containing:

- **conversion toolchain**: Docker image archive for ONNX-to-DXNN conversion
- **runtime library**: Shared library `libRuntimeLibrary.so` compiled for the target architectures and OS versions



> For details on how to build and use each component, refer to the respective READMEs:  
> - [Conversion Toolchain Guide](conversion-toolchain/README.md)  
> - [Runtime Library Guide](runtime-library/README.md)

---

## Compatibility

- **Supported Chip**: DX-M1 
- **Required DeepX Runtime (DX-RT) version**: Must match the version used during the build process

> The runtime-library is built inside a Docker environment that mimics the target architecture and OS version, with DeepX Runtime (DX-RT) preinstalled. Therefore, the resulting `libRuntimeLibrary.so` must be used in a runtime environment that matches the same architecture, OS version, and has the same DX-RT version installed.

---

## Limitations

- Currently, **only Ubuntu-based builds are supported** (20.04, 22.04, and 24.04).
- **Windows build is not supported** at this time.

---

## License

This project is licensed under the **Apache License 2.0**.  
See the [LICENSE](LICENSE) file for full license text.