#!/usr/bin/env bash
# Launch TTS server via PowerShell so it runs as a native Windows process,
# avoiding VS Code auto-port-forwarding (which intercepts WSL-spawned processes).
#
# Usage (from bash tool or terminal):
#   ./start_server.sh
#   ./start_server.sh 8765 10000

PORT=${1:-8765}
VRAM=${2:-10000}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe \
    -ExecutionPolicy Bypass \
    -File "$(cygpath -w "$SCRIPT_DIR/start_server.ps1")" \
    -Port "$PORT" \
    -VramMb "$VRAM"
