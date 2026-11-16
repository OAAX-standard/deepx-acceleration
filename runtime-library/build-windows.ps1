<#
.SYNOPSIS
    Builds RuntimeLibrary.dll for Windows x64 and packages it like the Linux build.

.DESCRIPTION
    DX-RT comes from one of:
      -DxrtRoot <dir>  An installed DX-RT SDK: a directory with include\dxrt\, lib\dxrt.lib and
                       lib\cmake\dxrt\, such as the DX-SDK Windows Installer's
                       C:\Program Files\DEEPX\M1_M1M_Series\DX-RUNTIME\<version>.
      (omitted)        The public dx_rt source in runtime-library\dx_rt, which is built and installed
                       to dx_rt\out\install\x64-Release first (what CI does). Get it with
                       git clone --branch v<dxrt_version.txt> https://github.com/DEEPX-AI/dx_rt.git runtime-library\dx_rt

    RuntimeLibrary.dll links dxrt.dll dynamically and does not ship it: at run time it uses the
    dxrt.dll the DX-SDK Windows Installer put on PATH.

    Output in artifacts\x86_64-windows\: RuntimeLibrary.dll, its import library RuntimeLibrary.lib
    (to link tests\runtime against it), and library-dxrt-<DX-RT version>.tar.gz holding the DLL only.

.EXAMPLE
    .\build-windows.ps1 -DxrtRoot "C:\Program Files\DEEPX\M1_M1M_Series\DX-RUNTIME\2.5.0"

.EXAMPLE
    .\build-windows.ps1
#>
[CmdletBinding()]
param(
    [ValidateSet("Release", "Debug")]
    [string]$Configuration = "Release",
    [string]$DxrtRoot,
    [string]$RuntimeVersion
)

$ErrorActionPreference = "Stop"

# Runs a native command and throws if it fails.  Its output goes to the console, never into the
# caller's return value.
function Invoke-Native([string]$What, [scriptblock]$Command) {
    & $Command | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "$What failed with exit code $LASTEXITCODE." }
}

# Loads the Visual Studio 2022 x64 developer environment (cl, Ninja, VCPKG_ROOT) into this process.
function Import-VsDevEnvironment {
    if ($env:VSCMD_VER) { return }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    $vsPath = $null
    if (Test-Path $vswhere) {
        $vsPath = & $vswhere -latest -products * -version "[17.0,18.0)" `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    }
    if (-not $vsPath) { throw "Visual Studio 2022 with the C++ x64 build tools is required." }
    $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
    $environment = cmd /c "`"$vcvars`" >nul && set"
    if ($LASTEXITCODE -ne 0) { throw "$vcvars failed with exit code $LASTEXITCODE." }
    foreach ($line in $environment) {
        if ($line -match "^([^=]+)=(.*)$") { Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2] }
    }
}

# Builds and installs the public dx_rt source with its own CMake preset and returns the install
# directory.  dx_rt's build.ps1 is not used: besides building, it stops the dxrtd service and
# reinstalls dx_rt's Python package into every Python on the machine.
function Build-DxrtSource {
    $source = Join-Path $PSScriptRoot "dx_rt"
    if (-not (Test-Path (Join-Path $source "CMakePresets.json"))) {
        $tag = "v" + (Get-Content (Join-Path $PSScriptRoot "dxrt_version.txt") -Raw).Trim()
        throw "DX-RT source not found in $source. Pass -DxrtRoot, or clone it: " +
            "git clone --branch $tag https://github.com/DEEPX-AI/dx_rt.git `"$source`""
    }
    Import-VsDevEnvironment
    if (-not $env:VCPKG_ROOT) { throw "VCPKG_ROOT is not set; dx_rt's CMake preset needs vcpkg." }
    # Like dx_rt's build.ps1: keep vcpkg downloads out of the (possibly read-only) vcpkg root.
    if (-not $env:VCPKG_DOWNLOADS) { $env:VCPKG_DOWNLOADS = Join-Path $source "out\vcpkg-downloads" }

    $preset = "x64-Release"
    Push-Location $source
    try {
        Invoke-Native "DX-RT configure" { cmake --preset $preset }
        Invoke-Native "DX-RT build" { cmake --build --preset $preset }
        Invoke-Native "DX-RT install" { cmake --install "out\build\$preset" }
    } finally {
        Pop-Location
    }
    return Join-Path $source "out\install\$preset"
}

# Returns the DXRT_VERSION of the SDK (e.g. "v3.5.0"), which names the archive as on Linux.
function Get-DxrtVersion([string]$Root) {
    $genHeader = Join-Path $Root "include\dxrt\gen.h"
    if (Test-Path $genHeader) {
        $match = Select-String -Path $genHeader -Pattern '#define\s+DXRT_VERSION\s+"([^"]+)"' | Select-Object -First 1
        if ($match) { return $match.Matches[0].Groups[1].Value }
    }
    throw "Cannot read DXRT_VERSION from $genHeader."
}

if (-not $RuntimeVersion) {
    $RuntimeVersion = (Get-Content (Join-Path $PSScriptRoot "..\VERSION") -Raw).Trim()
}

if ($DxrtRoot) {
    $DxrtRoot = (Resolve-Path $DxrtRoot).Path
} else {
    $DxrtRoot = Build-DxrtSource
}
foreach ($required in "include\dxrt\dxrt_cxx_api.h", "lib\dxrt.lib", "lib\cmake\dxrt") {
    if (-not (Test-Path (Join-Path $DxrtRoot $required))) { throw "$DxrtRoot is not a DX-RT SDK: $required is missing." }
}
$dxrtVersion = Get-DxrtVersion $DxrtRoot
$pinnedVersion = "v" + (Get-Content (Join-Path $PSScriptRoot "dxrt_version.txt") -Raw).Trim()
if ($dxrtVersion -ne $pinnedVersion) {
    Write-Warning "DX-RT in $DxrtRoot is $dxrtVersion; dxrt_version.txt pins $pinnedVersion."
}
Write-Host "DX-RT $dxrtVersion from $DxrtRoot"

# Always configure from scratch: a cached dxrt_DIR would keep linking a previously used SDK.
$buildDir = Join-Path $PSScriptRoot "build-windows"
if (Test-Path $buildDir) { Remove-Item -Recurse -Force $buildDir }
Invoke-Native "CMake configure" {
    cmake -S $PSScriptRoot -B $buildDir -G "Visual Studio 17 2022" -A x64 `
        "-DDXRT_ROOT=$DxrtRoot" "-DOAAX_RUNTIME_VERSION=$RuntimeVersion"
}
Invoke-Native "Build" { cmake --build $buildDir --config $Configuration --parallel }

# Same layout as the Linux build: artifacts\<arch>-<os>\
$artifactsDir = Join-Path $PSScriptRoot "artifacts\x86_64-windows"
if (Test-Path $artifactsDir) { Remove-Item -Recurse -Force $artifactsDir }
New-Item -ItemType Directory -Path $artifactsDir | Out-Null
$outputDir = Join-Path $buildDir $Configuration
Copy-Item (Join-Path $outputDir "RuntimeLibrary.dll"), (Join-Path $outputDir "RuntimeLibrary.lib") $artifactsDir

$archiveName = "library-dxrt-$dxrtVersion.tar.gz"
Invoke-Native "tar" { tar -czf (Join-Path $artifactsDir $archiveName) -C $artifactsDir RuntimeLibrary.dll }
Write-Host "Built $artifactsDir\RuntimeLibrary.dll and $archiveName"
