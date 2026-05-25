param(
    [string]$ProjectRoot = "",
    [switch]$SkipGpuContainerTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Text)
    Write-Host ""
    Write-Host "== $Text ==" -ForegroundColor Cyan
}

function Write-Pass {
    param([string]$Text)
    Write-Host "[PASS] $Text" -ForegroundColor Green
}

function Write-Fail {
    param([string]$Text)
    Write-Host "[FAIL] $Text" -ForegroundColor Red
}

function Write-Warn {
    param([string]$Text)
    Write-Host "[WARN] $Text" -ForegroundColor Yellow
}

function Command-Exists {
    param([string]$CommandName)
    return [bool](Get-Command $CommandName -ErrorAction SilentlyContinue)
}

$failures = New-Object System.Collections.Generic.List[string]
$warnings = New-Object System.Collections.Generic.List[string]

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
} else {
    $ProjectRoot = (Resolve-Path $ProjectRoot).Path
}

Write-Section "Windows + WSL + Docker GPU Preflight"
Write-Host "Project root: $ProjectRoot"

$runningOnWindows = $false
if (Get-Variable -Name IsWindows -Scope Global -ErrorAction SilentlyContinue) {
    $runningOnWindows = [bool]$IsWindows
} else {
    $runningOnWindows = ($PSVersionTable.PSEdition -eq "Desktop") -or ($env:OS -eq "Windows_NT")
}

if (-not $runningOnWindows) {
    Write-Fail "This script is for Windows hosts. Current platform is not Windows."
    exit 1
}
Write-Pass "Running on Windows"

Write-Section "Command Availability"
foreach ($cmd in @("wsl", "docker")) {
    if (Command-Exists -CommandName $cmd) {
        Write-Pass "Found '$cmd'"
    } else {
        Write-Fail "Missing '$cmd' in PATH"
        $failures.Add("Command '$cmd' is not available.")
    }
}

if ($failures.Count -gt 0) {
    Write-Host ""
    Write-Fail "Preflight stopped due to missing commands."
    exit 1
}

Write-Section "WSL2 Checks"
try {
    $wslStatus = (& wsl --status) | Out-String
    Write-Pass "WSL is installed"
    if ($wslStatus -match "Default Version:\s*2") {
        Write-Pass "WSL default version is 2"
    } else {
        Write-Warn "Could not confirm 'Default Version: 2' from 'wsl --status'."
        $warnings.Add("Set WSL default version to 2 using: wsl --set-default-version 2")
    }
} catch {
    Write-Fail "Failed to run 'wsl --status': $($_.Exception.Message)"
    $failures.Add("WSL status command failed.")
}

try {
    & wsl --update | Out-Null
    Write-Pass "WSL kernel update command succeeded"
} catch {
    Write-Warn "Could not run 'wsl --update': $($_.Exception.Message)"
    $warnings.Add("Run 'wsl --update' manually in elevated PowerShell.")
}

Write-Section "Docker Desktop Checks"
try {
    $dockerServerVersion = (& docker version --format '{{.Server.Version}}' 2>$null).Trim()
    if ([string]::IsNullOrWhiteSpace($dockerServerVersion)) {
        Write-Fail "Docker daemon is not responding."
        $failures.Add("Docker daemon unavailable.")
    } else {
        Write-Pass "Docker daemon reachable (Server $dockerServerVersion)"
    }
} catch {
    Write-Fail "Docker daemon check failed: $($_.Exception.Message)"
    $failures.Add("Docker daemon unavailable.")
}

try {
    $dockerOsType = (& docker info --format '{{.OSType}}' 2>$null).Trim()
    if ($dockerOsType -eq "linux") {
        Write-Pass "Docker is using Linux containers"
    } else {
        Write-Fail "Docker OSType is '$dockerOsType' (expected 'linux')."
        $failures.Add("Switch Docker Desktop to Linux containers.")
    }
} catch {
    Write-Fail "Failed to inspect Docker OSType: $($_.Exception.Message)"
    $failures.Add("Could not confirm Linux containers mode.")
}

Write-Section "NVIDIA Driver Checks"
if (Command-Exists -CommandName "nvidia-smi") {
    try {
        & nvidia-smi | Out-Null
        Write-Pass "Host NVIDIA driver is available (nvidia-smi ok)"
    } catch {
        Write-Warn "nvidia-smi exists but failed to run."
        $warnings.Add("Check NVIDIA driver installation.")
    }
} else {
    Write-Warn "nvidia-smi not found on host PATH."
    $warnings.Add("Install/update NVIDIA GPU driver for Windows + WSL2 support.")
}

Write-Section "Project Files"
$requiredPaths = @(
    (Join-Path $ProjectRoot "Dockerfile.rtx"),
    (Join-Path $ProjectRoot "requirements-docker.txt"),
    (Join-Path $ProjectRoot "config.yaml")
)
foreach ($path in $requiredPaths) {
    if (Test-Path $path) {
        Write-Pass "Found $(Split-Path $path -Leaf)"
    } else {
        Write-Fail "Missing required file: $path"
        $failures.Add("Missing file: $path")
    }
}

$composeCandidates = @(
    (Join-Path $ProjectRoot "docker-compose.rtx.yml"),
    (Join-Path $ProjectRoot "docker-compose.gpu.yml")
)
$composeFound = $false
foreach ($path in $composeCandidates) {
    if (Test-Path $path) {
        Write-Pass "Found compose file $(Split-Path $path -Leaf)"
        $composeFound = $true
    }
}
if (-not $composeFound) {
    Write-Fail "Missing RTX compose file (expected docker-compose.rtx.yml or docker-compose.gpu.yml)."
    $failures.Add("Missing compose file for RTX run.")
}

$modelsPath = Join-Path $ProjectRoot "models"
if (Test-Path $modelsPath) {
    Write-Pass "Found models directory"
} else {
    Write-Warn "Missing models directory at $modelsPath"
    $warnings.Add("Create/mount models directory before runtime.")
}

$videosPath = Join-Path $ProjectRoot "videos"
if (Test-Path $videosPath) {
    Write-Pass "Found videos directory"
} else {
    Write-Warn "Missing videos directory at $videosPath"
    $warnings.Add("Create/mount videos directory before runtime.")
}

Write-Section "GPU Container Test"
if ($SkipGpuContainerTest) {
    Write-Warn "Skipped GPU container test (--SkipGpuContainerTest)"
} else {
    $cudaImage = "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04"
    try {
        & docker run --rm --gpus all $cudaImage nvidia-smi | Out-Null
        Write-Pass "GPU is accessible inside Docker container"
    } catch {
        Write-Fail "GPU container test failed."
        $failures.Add("Docker GPU test failed. Verify Docker Desktop WSL2 GPU support and NVIDIA setup.")
    }
}

Write-Section "Summary"
if ($warnings.Count -gt 0) {
    Write-Warn ("Warnings: " + $warnings.Count)
    foreach ($w in $warnings) {
        Write-Warn " - $w"
    }
}

if ($failures.Count -gt 0) {
    Write-Fail ("Failures: " + $failures.Count)
    foreach ($f in $failures) {
        Write-Fail " - $f"
    }
    exit 1
}

Write-Pass "All critical checks passed."
Write-Host ""
Write-Host "Next:"
Write-Host "  1) docker compose -f docker-compose.gpu.yml up --build -d"
Write-Host "  2) curl http://localhost:8000/health"
Write-Host "  3) curl http://localhost:8000/api/runtime/acceleration"
