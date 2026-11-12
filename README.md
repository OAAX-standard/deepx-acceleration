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
- **OAAX version**: 2.0 (callback-based C API, see [`runtime-library/include/runtime_core.h`](runtime-library/include/runtime_core.h))
- **Required DX-RT version**: Must match the version used during the runtime-library build process. The version used by this repo is pinned in `runtime-library/dxrt_version.txt` (Linux and Windows)

> On Linux, the runtime-library is built inside a Docker environment that mimics the target architecture and OS version, with DeepX Runtime (DX-RT) preinstalled. Therefore, the resulting `libRuntimeLibrary.so` must be used in a runtime environment that matches the same architecture, OS version, and has the same DX-RT version installed.
>
> On Windows, the runtime-library is built natively with Visual Studio 2022 against the same DX-RT version (the DX-SDK Windows Installer's SDK or the public `dx_rt` source). `RuntimeLibrary.dll` does not bundle `dxrt.dll`: it uses the one installed by the DX-SDK Windows Installer (1.3.2 or later for DX-RT 3.5.0).

---

## Testing

- **Unit tests** (no hardware, run in CI): `pytest tests/test_models.py` and the mock-DX-RT runtime tests in [`runtime-library/test`](runtime-library/README.md#unit-tests).
- **Conversion + inference tests** (Docker, and DeepX hardware for inference): see the [test suite guide](tests/README.md).
- **Linting**: `pre-commit run --all-files` (ruff, clang-format, shellcheck, hadolint) — the same checks CI runs.

---

## Limitations

- **Linux builds**: Currently supported for Ubuntu-based targets (20.04, 22.04, and 24.04).
- **Windows builds**: x86_64 only, built with Visual Studio 2022 (`runtime-library/build-windows.ps1`).

---

## Contributing

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). Every commit needs a DCO sign-off (`git commit -s`).

---

## License

This project is licensed under the **Apache License 2.0**.  
See the [LICENSE](LICENSE) file for full license text. Vendored third-party code is listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
DX-RT and DX-COM are proprietary DEEPX software and are not covered by this license.
