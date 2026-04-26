param(
    [string]$Config = "configs/eval_quality.yaml"
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
& $venvPython -u -m distill_qwen.evaluate_quality --config $Config
Pop-Location
