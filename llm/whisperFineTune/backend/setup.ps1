$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
  throw "Node.js is required. Install Node.js 22.5 or newer, then run setup again."
}

$nodeVersion = [version]((node --version).TrimStart('v'))
if ($nodeVersion -lt [version]'22.5.0') {
  throw "Node.js 22.5 or newer is required for this package. Found: $(node --version)"
}

if (-not (Test-Path -LiteralPath ".env")) {
  Copy-Item -LiteralPath ".env.example" -Destination ".env"
  Write-Host "Created .env from .env.example"
}

New-Item -ItemType Directory -Force -Path ".\data" | Out-Null
npm install

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$projectPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $projectPython) {
  Write-Host "Found project Python environment: $projectPython"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
  $localVenv = Join-Path $PSScriptRoot ".venv"
  $localPython = Join-Path $localVenv "Scripts\python.exe"
  if (-not (Test-Path -LiteralPath $localPython)) {
    Write-Host "Creating local Python environment..."
    & py -3.11 -m venv $localVenv
  }
  if (-not (Test-Path -LiteralPath $localPython)) {
    throw "Could not create the local Python environment. Install Python 3.11 and run setup again."
  }
  & $localPython -m pip install --upgrade pip
  & $localPython -m pip install -r ".\python\requirements.txt"
  Write-Host "Installed Python worker dependencies into: $localVenv"
} else {
  Write-Warning "Python was not found. Install Python 3.11+ before starting the Whisper worker."
}

Write-Host "Setup complete. Run .\start.ps1 to start the backend and GPU worker."
