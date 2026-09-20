$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$env:HF_HOME = Join-Path $projectRoot '.cache\huggingface'
$env:MPLCONFIGDIR = Join-Path $projectRoot '.cache\matplotlib'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$logDir = Join-Path $projectRoot 'experiments\llama_logs'
New-Item -ItemType Directory -Force $logDir | Out-Null
$statusFile = Join-Path $logDir 'status.txt'

function Set-RunStatus([string]$message) {
    Set-Content -LiteralPath $statusFile -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - $message"
}

function Invoke-Stage([string]$name, [string[]]$arguments) {
    Set-RunStatus "$name running."
    $stdout = Join-Path $logDir "$name.log"
    $stderr = Join-Path $logDir "$name.stderr.log"
    $ErrorActionPreference = 'Continue'
    & $python -u @arguments > $stdout 2> $stderr
    $stageExitCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($stageExitCode -ne 0) {
        throw "$name failed with exit code $stageExitCode. See $stdout and $stderr."
    }
    Set-RunStatus "$name complete."
}

try {
    Invoke-Stage 'baseline' @('scripts/run_baseline.py', '--config', 'configs/llama3_1b.json')
    Invoke-Stage 'measurement' @('scripts/run_measurement.py', '--config', 'configs/llama3_1b_measurement.json')
    Invoke-Stage 'pruning' @('scripts/run_pruned.py', '--config', 'configs/llama3_1b.json')
    Set-RunStatus 'Complete: Llama baseline, full measurement, lowest-BI sweep, and three random sweeps.'
} catch {
    Set-RunStatus "Stopped: $_"
    throw
}
