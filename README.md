# DEEPX OAAX Implementation


<img src="https://cdn.prod.website-files.com/6649422e9fd6f89e2b293704/6649422e9fd6f89e2b293710_logo.svg" alt="OAAX" width="49%" />
<img src="./docs/images/image.png" alt="DeepX Acceleration" width="49%" />

This repository contains the source code for building the implementation of the **conversion toolchain and runtime library for DeepX NPU acceleration**, based on the **OAAX standard**. 
> If you're new to OAAX, you can find more information about the OAAX standard in the [OAAX repository](https://github.com/oaax-standard/OAAX)

This project builds on top of the [reference implementation](https://github.com/OAAX-standard/reference-implementation) provided in the OAAX repository.
Please check it out first to gain a high-level understanding of how an OAAX-compliant runtime and conversion toolchain should be implemented.

---

## Getting Started

To clone the repository:

```bash
git clone https://github.com/OAAX-standard/deepx-acceleration.git
```

---

## Repository structure

The repository is organized into two main components:

- **[`conversion-toolchain`](conversion-toolchain/)**:
Contains the scripts and Docker setup for converting ONNX models into the DXNN (DeepX Neural Network) format using the DeepX Compiler (`DX-COM`).

- **[`runtime-library`](runtime-library/)**:
Provides the source code and build system (Docker-based for Linux, native CMake/Visual Studio on Windows) to generate the C++ runtime library (`libRuntimeLibrary.so` / `RuntimeLibrary.dll`), which is responsible for loading and executing DXNN models on DeepX NPUs using the DeepX Runtime (`DX-RT`).

---

## Building the implementation

You can build both the **conversion-toolchain** and the **runtime-library** independently using the provided shell scripts in each folder.  
Upon successful build, an `artifacts/` directory will be created in each component, containing:

- **conversion-toolchain**: Docker image archive for ONNX-to-DXNN conversion
- **runtime-library**: Shared libraries compiled for the target architectures and operating systems
	- Linux targets: `libRuntimeLibrary.so`
	- Windows targets: `RuntimeLibrary.dll`



> For details on how to build and use each component, refer to the respective READMEs:  
> - [Conversion Toolchain Guide](conversion-toolchain/README.md)  
> - [Runtime Library Guide](runtime-library/README.md)

---

## Compatibility

- **Supported Chips**: DX-M1, DX-H1 
- **Required DX-RT version**: Must match the version used during the runtime-library build process

> On Linux, the runtime-library is built inside a Docker environment that mimics the target architecture and OS version, with DeepX Runtime (DX-RT) preinstalled. Therefore, the resulting `libRuntimeLibrary.so` must be used in a runtime environment that matches the same architecture, OS version, and has the same DX-RT version installed.
>
> On Windows, the runtime-library is built natively using CMake and Visual Studio against the DX-RT Windows SDK. The resulting `RuntimeLibrary.dll` must be used with a matching DX-RT Windows SDK version.

---

## Limitations

- **Linux builds**: Currently supported for Ubuntu-based targets (20.04, 22.04, and 24.04).
- **Windows builds**: Supported for x86_64 using Visual Studio 2022 and the DX-RT Windows SDK.

---

## License

This project is licensed under the **Apache License 2.0**.  
See the [LICENSE](LICENSE) file for full license text.