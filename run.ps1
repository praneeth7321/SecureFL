$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

function Run-Step {
    param([string]$Description, [scriptblock]$Command)
    Write-Host "`n$Description" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed (exit code $LASTEXITCODE)."
    }
}

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw 'Install Python 3.12 from python.org and enable the Python launcher (py), then run this file again.'
}
Run-Step 'Checking Python 3.12...' { py -3.12 -c 'import sys; print(sys.version)' }

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Run-Step 'Creating the project environment...' { py -3.12 -m pip install --user uv }
    Run-Step 'Creating .venv...' { py -3.12 -m uv venv .venv --python 3.12 }
}

& $python -c 'import flwr, torch, sklearn, pandas, matplotlib, ray'
if ($LASTEXITCODE -ne 0) {
    Run-Step 'Installing project dependencies (this can take a few minutes)...' {
        py -3.12 -m pip install --user uv
        if ($LASTEXITCODE -ne 0) { throw 'Could not install uv.' }
        py -3.12 -m uv pip install --python $python --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple --index-strategy unsafe-best-match -r requirements.txt
    }
}

$ton = Join-Path $PSScriptRoot 'data\ton_iot\train_test_network.csv'
$cicTrain = Join-Path $PSScriptRoot 'data\ciciot2023\train\train.csv'
$cicTest = Join-Path $PSScriptRoot 'data\ciciot2023\test\test.csv'
if (-not (Test-Path $ton) -or -not (Test-Path $cicTrain) -or -not (Test-Path $cicTest)) {
    $downloads = Join-Path $HOME 'Downloads'
    $tonArchive = Join-Path $downloads 'archive.zip'
    $cicArchive = Join-Path $downloads 'archive(1).zip'
    if (-not (Test-Path $ton) -and -not (Test-Path $tonArchive)) {
        throw "TON_IoT data is missing. Place archive.zip in $downloads and run again."
    }
    $prepareArgs = @('scripts\prepare_data.py')
    if (Test-Path $tonArchive) { $prepareArgs += @('--ton-archive', $tonArchive) }
    if (Test-Path $cicArchive) { $prepareArgs += @('--cic-archive', $cicArchive) }
    Run-Step 'Preparing dataset files...' { & $python @prepareArgs }
}

$env:MPLBACKEND = 'Agg'
Run-Step 'Running SecureFL on TON_IoT and CICIoT2023...' {
    & $python main.py --data-dir data --output-dir results
}
Write-Host "`nDone. Tables and graphs are in $PSScriptRoot\results" -ForegroundColor Green
