<#
.SYNOPSIS
    Builds inference_runner.exe against RuntimeLibrary.dll (Windows counterpart of build-tests.sh).

.DESCRIPTION
    -RuntimeLibDir is a directory with RuntimeLibrary.dll and its import library RuntimeLibrary.lib;
    it defaults to runtime-library\artifacts\x86_64-windows, the output of
    runtime-library\build-windows.ps1.  The DLL is copied next to inference_runner.exe.  dxrt.dll is
    loaded from PATH: install the DX-SDK Windows Installer before running the binary.

.EXAMPLE
    .\tests\runtime\build-tests.ps1
    .\tests\runtime\build\Release\inference_runner.exe --model model.dxnn --input-size 150528
#>
[CmdletBinding()]
param(
    [string]$RuntimeLibDir,
    [string]$BuildDir
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $RuntimeLibDir) { $RuntimeLibDir = Join-Path $repoRoot "runtime-library\artifacts\x86_64-windows" }
if (-not $BuildDir) { $BuildDir = Join-Path $PSScriptRoot "build" }

foreach ($file in "RuntimeLibrary.dll", "RuntimeLibrary.lib") {
    if (-not (Test-Path (Join-Path $RuntimeLibDir $file))) {
        throw "$file not found in $RuntimeLibDir. Build the runtime library first: runtime-library\build-windows.ps1"
    }
}
Write-Host "Using runtime library from: $RuntimeLibDir"

cmake -S $PSScriptRoot -B $BuildDir -G "Visual Studio 17 2022" -A x64 `
    "-DRUNTIME_LIB_DIR=$RuntimeLibDir" "-DRUNTIME_INCLUDE_DIR=$repoRoot\runtime-library\include"
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed with exit code $LASTEXITCODE." }
cmake --build $BuildDir --config Release
if ($LASTEXITCODE -ne 0) { throw "Build failed with exit code $LASTEXITCODE." }

Write-Host ""
Write-Host "Build complete: $BuildDir\Release\inference_runner.exe"
