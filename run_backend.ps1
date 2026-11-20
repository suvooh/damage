$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

# Prefer a local venv over the system Python.
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

$port = if ($env:DAMAGE_API_PORT) { $env:DAMAGE_API_PORT } elseif ($env:VEHICLE_DAMAGE_API_PORT) { $env:VEHICLE_DAMAGE_API_PORT } else { "8001" }
$device = if ($env:VEHICLE_DAMAGE_DEVICE) { $env:VEHICLE_DAMAGE_DEVICE } else { "auto" }

$env:PYTHONPATH = Join-Path $projectRoot "src"
Write-Host "Starting the damage detection API on http://127.0.0.1:$port ..."
Write-Host "Model device: $device (set VEHICLE_DAMAGE_DEVICE=gpu|npu|cpu|cuda to override)"
& $python -m vehicle_damage.api
