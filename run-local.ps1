# Run from any directory; keep downloaded models and caches inside the project.
$ErrorActionPreference = 'Stop'
$env:HF_HOME = Join-Path $PSScriptRoot '.cache\huggingface'
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot '.tools\python'
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.cache\matplotlib'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
Push-Location $PSScriptRoot
try {
    & "$PSScriptRoot\.venv\Scripts\python.exe" @args
    $result = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $result
