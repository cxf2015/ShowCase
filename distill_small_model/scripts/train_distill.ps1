param(
    [string]$Config = "configs/distill_qwen_0.2b.yaml",
    [switch]$SkipGenerate
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

if (-not $SkipGenerate) {
    & $venvPython -m distill_qwen.generate_distill_data --config $Config
    if ($LASTEXITCODE -ne 0) {
        throw "Teacher data generation failed. Training has been stopped."
    }
}

& $venvPython -m distill_qwen.train_student --config $Config
if ($LASTEXITCODE -ne 0) {
    throw "Student training failed."
}

Pop-Location
