$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path -LiteralPath ".\node_modules")) {
  throw "Dependencies are missing. Run .\setup.ps1 first."
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$projectPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$localPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $projectPython) -and -not (Test-Path -LiteralPath $localPython)) {
  throw "Python worker environment is missing. Run .\setup.ps1 first."
}

$port = 3000
$envFile = Join-Path $PSScriptRoot ".env"
if (Test-Path -LiteralPath $envFile) {
  $portLine = Get-Content -LiteralPath $envFile | Where-Object { $_ -match '^\s*PORT\s*=' } | Select-Object -First 1
  if ($portLine -match '=\s*(\d+)') { $port = [int]$Matches[1] }
}

$existingListener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($existingListener) {
  try {
    $health = Invoke-RestMethod "http://127.0.0.1:$port/api/health" -TimeoutSec 3
    if ($health.service -eq "whisper-finetune-local-backend") {
      Write-Host "Backend is already running at http://127.0.0.1:$port"
      Write-Host "Open that address in your browser. No second server was started."
      exit 0
    }
  } catch {
    throw "Port $port is already in use by another application. Change PORT in .env or stop that application."
  }
  throw "Port $port is already in use by another application. Change PORT in .env or stop that application."
}

npm start
