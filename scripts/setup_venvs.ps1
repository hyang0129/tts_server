# setup_venvs.ps1 - Create and populate per-engine venvs for tts_server on Windows.
#
# Venv layout (all relative to repo root):
#   .venv\               - host (FastAPI/uvicorn only, no engine packages)
#   .venvs\chatterbox\   - Chatterbox Turbo + Full engines
#   .venvs\higgs\        - Higgs Audio engine (transformers<4.47.0)
#   .venvs\qwen3\        - Qwen3-TTS engine (transformers>=4.57.3)
#
# Venvs are created sequentially (fast), then all pip installs run in parallel.
# Idempotent: skips venv creation if directory exists, but always re-runs pip install.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_venvs.ps1
#
# Requirements:
#   Python 3.11 available as "py -3.11"
#   NVIDIA GPU (Blackwell/sm_12x auto-detected for cu128 torch override)

param(
    [string]$HiggsRepoPath = "D:\tmp\faster-higgs-audio"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot

# Detect Blackwell GPU (sm_12.x) - needs cu128 torch
$IsBlackwell = $false
try {
    $cap = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>&1) | Out-String
    if ($cap -match "^12\.") { $IsBlackwell = $true }
} catch {}

Write-Host "==> tts_server venv setup (Windows, parallel installs)"
Write-Host "    Repo root    : $Repo"
Write-Host "    Blackwell GPU: $IsBlackwell"
Write-Host ""

# ---------------------------------------------------------------------------
# Step 1: Create venvs (sequential, fast)
# ---------------------------------------------------------------------------
function New-VenvIfMissing {
    param([string]$VenvPath)
    if (Test-Path $VenvPath) {
        Write-Host "[skip]   Venv already exists: $VenvPath"
    } else {
        Write-Host "[create] $VenvPath"
        & py -3.11 -m venv $VenvPath
        Write-Host "[ok]     Created $VenvPath"
    }
}

$HostVenv = Join-Path $Repo ".venv"
$CbVenv   = Join-Path $Repo ".venvs\chatterbox"
$HiggsVenv = Join-Path $Repo ".venvs\higgs"
$Qwen3Venv = Join-Path $Repo ".venvs\qwen3"

Write-Host "--- Creating venvs ---"
New-VenvIfMissing -VenvPath $HostVenv
New-VenvIfMissing -VenvPath $CbVenv
New-VenvIfMissing -VenvPath $HiggsVenv
New-VenvIfMissing -VenvPath $Qwen3Venv
Write-Host ""

# Clone faster-higgs-audio if not already present (do this before parallel jobs start)
if (Test-Path $HiggsRepoPath) {
    Write-Host "[skip]   faster-higgs-audio already at $HiggsRepoPath"
} else {
    Write-Host "[clone]  Cloning faster-higgs-audio to $HiggsRepoPath"
    git clone https://github.com/sorbetstudio/faster-higgs-audio $HiggsRepoPath
    Write-Host "[ok]     Cloned"
}
Write-Host ""

# ---------------------------------------------------------------------------
# Step 2: Install deps in parallel (one job per venv)
# ---------------------------------------------------------------------------
Write-Host "--- Installing deps in parallel ---"

$jobs = @()

# Host job
$jobs += Start-Job -Name "host" -ScriptBlock {
    param($VenvPath, $ReqsFile)
    $pip = Join-Path $VenvPath "Scripts\pip.exe"
    & $pip install --upgrade pip --quiet 2>&1 | Out-Null
    & $pip install -r $ReqsFile 2>&1
} -ArgumentList $HostVenv, (Join-Path $Repo "requirements-host.txt")

# Chatterbox job
$jobs += Start-Job -Name "chatterbox" -ScriptBlock {
    param($VenvPath, $ReqsFile, $IsBlackwell)
    $pip = Join-Path $VenvPath "Scripts\pip.exe"
    & $pip install --upgrade pip --quiet 2>&1 | Out-Null
    & $pip install -r $ReqsFile 2>&1
    if ($IsBlackwell) {
        & $pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128 --quiet 2>&1
    }
} -ArgumentList $CbVenv, (Join-Path $Repo "requirements-chatterbox.txt"), $IsBlackwell

# Higgs job
$jobs += Start-Job -Name "higgs" -ScriptBlock {
    param($VenvPath, $ReqsFile, $IsBlackwell)
    $pip = Join-Path $VenvPath "Scripts\pip.exe"
    & $pip install --upgrade pip --quiet 2>&1 | Out-Null
    & $pip install -r $ReqsFile 2>&1
    if ($IsBlackwell) {
        & $pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128 --quiet 2>&1
    }
} -ArgumentList $HiggsVenv, (Join-Path $Repo "requirements-higgs.txt"), $IsBlackwell

# Qwen3 job
$jobs += Start-Job -Name "qwen3" -ScriptBlock {
    param($VenvPath, $ReqsFile, $IsBlackwell)
    $pip = Join-Path $VenvPath "Scripts\pip.exe"
    & $pip install --upgrade pip --quiet 2>&1 | Out-Null
    & $pip install -r $ReqsFile 2>&1
    if ($IsBlackwell) {
        & $pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128 --quiet 2>&1
    }
} -ArgumentList $Qwen3Venv, (Join-Path $Repo "requirements-qwen3.txt"), $IsBlackwell

Write-Host "Jobs started: $($jobs.Name -join ', ')"
Write-Host "Waiting for all installs to complete..."
Write-Host ""

# Wait and stream progress
while ($jobs | Where-Object { $_.State -eq 'Running' }) {
    $running = ($jobs | Where-Object { $_.State -eq 'Running' }).Name -join ', '
    Write-Host "  still running: $running"
    Start-Sleep -Seconds 10
}

Wait-Job -Job $jobs | Out-Null

# ---------------------------------------------------------------------------
# Step 3: Report results
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "--- Results ---"
$anyFailed = $false
foreach ($job in $jobs) {
    $out = Receive-Job -Job $job 2>&1
    if ($job.State -eq 'Failed' -or ($out -match 'ERROR' -and $out -notmatch 'To modify pip')) {
        Write-Host "[FAIL] $($job.Name)"
        $out | Where-Object { $_ -match 'ERROR|error' -and $_ -notmatch 'To modify pip' } | ForEach-Object { Write-Host "  $_" }
        $anyFailed = $true
    } else {
        Write-Host "[ok]   $($job.Name)"
    }
    Remove-Job -Job $job
}

Write-Host ""
if ($anyFailed) {
    Write-Host "==> Some installs failed. Check output above."
    exit 1
} else {
    Write-Host "==> All venvs ready. Start with: .\start_server.ps1"
}
