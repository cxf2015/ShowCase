param(
    [string]$Config = "configs/smoke_test.yaml"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$srcPath = Join-Path $projectRoot "src"

if (-not (Test-Path $venvPython)) {
    throw "Virtual environment not found. Run scripts/setup_env.ps1 first."
}

Push-Location $projectRoot
$env:PYTHONPATH = $srcPath
& $venvPython -m distill_qwen.train_student --config $Config
Pop-Location
