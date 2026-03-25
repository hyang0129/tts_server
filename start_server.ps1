# start_server.ps1 — Launch the TTS server as a native Windows process.
#
# WHY THIS SCRIPT EXISTS:
#   When uvicorn is started from a bash/WSL shell (e.g. the Claude bash tool),
#   VS Code treats it as a remote process and auto-forwards the port via its own
#   listener. That forwarding is unreliable — health checks time out and the server
#   is effectively unreachable from Windows tools (curl.exe, browsers, other repos).
#
#   Starting uvicorn via PowerShell makes it a true Windows-native process bound
#   to 127.0.0.1, which VS Code does not intercept. Port 8765 is used because
#   VS Code has claimed 8000–8003 for its own forwarding.
#
# USAGE:
#   .\start_server.ps1                     # defaults: port 8765, 10000 MB VRAM
#   .\start_server.ps1 -Port 8765
#   .\start_server.ps1 -Port 8765 -VramMb 8000
#
# FOR CLAUDE (bash tool): use start_server.sh, which calls this script via PowerShell.

param(
    [int]$Port = 8765,
    [int]$VramMb = 10000
)

$Repo    = $PSScriptRoot
$Python  = Join-Path $Repo ".venv\Scripts\python.exe"
$LogOut  = Join-Path $Repo "tts_server.log"
$LogErr  = Join-Path $Repo "tts_server_err.log"

# Kill any existing Python process already listening on this port so we get a clean start.
$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    Where-Object { (Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue).ProcessName -like "python*" }
if ($existing) {
    Write-Host "Stopping existing server (PID $($existing.OwningProcess))..."
    Stop-Process -Id $existing.OwningProcess -Force
    Start-Sleep -Seconds 1
}

Write-Host "Starting TTS server on 127.0.0.1:$Port (VRAM: ${VramMb}MB)..."

# AVAILABLE_VRAM_MB tells the model manager how much GPU memory is safe to use.
# The .env file in the repo root is auto-loaded by the server for all other config
# (HIGGS_REPO_PATH, HIGGS_QUANT_BITS, model IDs, etc.).
$env:AVAILABLE_VRAM_MB = "$VramMb"

# Start uvicorn as a detached Windows process so this script can return after the
# health check without keeping a terminal open.
$proc = Start-Process -FilePath $Python `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$Port" `
    -WorkingDirectory $Repo `
    -RedirectStandardOutput $LogOut `
    -RedirectStandardError $LogErr `
    -NoNewWindow -PassThru

Write-Host "PID: $($proc.Id)"

# Poll the health endpoint until the server is ready (models lazy-load so startup is fast).
# Retry for up to 40 seconds before giving up.
$url   = "http://127.0.0.1:$Port/health"
$ready = $false
for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Seconds 2
    try {
        $resp = Invoke-WebRequest -Uri $url -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
        Write-Host "Server ready: $($resp.Content)"
        $ready = $true
        break
    } catch {}
}

if (-not $ready) {
    Write-Host "Server did not become ready — check tts_server_err.log"
    exit 1
}
