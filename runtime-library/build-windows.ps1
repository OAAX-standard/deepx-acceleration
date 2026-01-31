param(
    [string]$Configuration = "Release",
    [string]$DxrtRoot,
    [string]$RuntimeVersion
)

if (-not $DxrtRoot) {
    Write-Error "DxrtRoot is required (path to DX-RT Windows SDK root containing include/ and lib/)."
    exit 1
}

# Resolve OAAX runtime version
if (-not $RuntimeVersion) {
    $versionPath = Join-Path $PSScriptRoot "..\VERSION"
    if (Test-Path $versionPath) {
        $RuntimeVersion = (Get-Content $versionPath -Raw).Trim()
    }
}

if (-not $RuntimeVersion) {
    Write-Error "RuntimeVersion is required. Pass -RuntimeVersion or ensure dxrt_version.txt exists."
    exit 1
}

$buildDir = Join-Path $PSScriptRoot "build-windows"
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

# Configure
cmake -S $PSScriptRoot -B $buildDir `
      -G "Visual Studio 17 2022" -A x64 `
      -DDXRT_ROOT="$DxrtRoot" `
      -DOAAX_RUNTIME_VERSION="$RuntimeVersion" `
      -DCMAKE_BUILD_TYPE=$Configuration

if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configuration failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

# Build
cmake --build $buildDir --config $Configuration --parallel

if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

# Prepare artifacts directory (aligned with Linux build layout)
$artifactsRoot = Join-Path $PSScriptRoot "artifacts"
$targetArch = "x86_64"  # Current Windows build targets x64
$artifactsDirName = "$targetArch-windows"
$artifactsDir = Join-Path $artifactsRoot $artifactsDirName

if (-not (Test-Path $artifactsRoot)) {
    New-Item -ItemType Directory -Path $artifactsRoot | Out-Null
}

if (Test-Path $artifactsDir) {
    Remove-Item -Recurse -Force $artifactsDir
}
New-Item -ItemType Directory -Path $artifactsDir | Out-Null

# Locate built DLL
$dllName = "RuntimeLibrary.dll"
$dllSourcePath = Join-Path (Join-Path $buildDir $Configuration) $dllName
if (-not (Test-Path $dllSourcePath)) {
    Write-Error "Expected DLL not found at $dllSourcePath. Ensure the CMake target name matches '$dllName'."
    exit 1
}

Copy-Item $dllSourcePath -Destination $artifactsDir -Force

Write-Host "RuntimeLibrary build completed. DLL copied to: $artifactsDir\$dllName"