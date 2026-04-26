param(
    [string]$PythonExe = "python",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $projectRoot ".venv"

if ($Force -and (Test-Path $venvPath)) {
    Remove-Item $venvPath -Recurse -Force
}

if (-not (Get-Command $PythonExe -ErrorAction SilentlyContinue)) {
    throw "Python executable '$PythonExe' was not found. Install Python 3.10+ first."
}

if (-not (Test-Path $venvPath)) {
    & $PythonExe -m venv $venvPath
}

$venvPython = Join-Path $venvPath "Scripts\python.exe"

& $venvPython -m pip install --upgrade pip setuptools wheel

$hasNvidia = $null -ne (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue)

if ($hasNvidia) {
    & $venvPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
} else {
    & $venvPython -m pip install torch torchvision torchaudio
}

& $venvPython -m pip install -r (Join-Path $projectRoot "requirements.txt")

Write-Host "Environment is ready: $venvPython"
