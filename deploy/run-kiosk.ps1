#Requires -Version 5.1
param(
    [switch]$Lite,
    [switch]$Fullscreen,
    [switch]$World,
    [string]$Camera = "auto"
)

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Host "No .venv yet. Run: .\deploy\setup-pc.ps1"
    exit 1
}
$Ckpt = Join-Path $Root "models\best.pt"
if (-not (Test-Path $Ckpt)) {
    Write-Host "Missing models\best.pt. Copy the trained grader here first."
    exit 1
}
$WorldPt = Join-Path $Root "models\yolov8s-worldv2.pt"
if (-not $World -and -not (Test-Path $WorldPt)) {
    $Lite = $true
}
$argv = @("src\kiosk.py", "--camera", $Camera)
if ($Fullscreen) { $argv += "--fullscreen" }
if ($Lite) { $argv += "--lite" }
if ($World) { $argv += "--world" }
& $Py @argv
exit $LASTEXITCODE
