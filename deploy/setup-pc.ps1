#Requires -Version 5.1
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    python -m venv .venv
    $Py = Join-Path $Root ".venv\Scripts\python.exe"
}
$Req = Join-Path $Root "requirements.txt"
$Kiosk = Join-Path $Root "requirements-kiosk.txt"
if (-not (Test-Path $Req)) {
    $Req = $Kiosk
}
if (-not (Test-Path $Req)) {
    Write-Host "Missing requirements.txt or requirements-kiosk.txt"
    exit 1
}
& $Py -m pip install --upgrade pip
& $Py -m pip install -r $Req
Write-Host "PC venv ready. Run: .\deploy\run-kiosk.ps1"
