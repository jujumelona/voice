param(
    [switch]$SkipFFmpeg,
    [switch]$SkipSampleDownload
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if ($SkipFFmpeg) {
    & "$root\scripts\setup_runtime.ps1" -SkipFFmpeg
} else {
    & "$root\scripts\setup_runtime.ps1"
}
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

New-Item -ItemType Directory -Force -Path "$root\test_input" | Out-Null
New-Item -ItemType Directory -Force -Path "$root\test_output" | Out-Null

if (-not $SkipSampleDownload) {
    $python = Join-Path $root ".runtime\python311\python.exe"
    & $python "$root\scripts\prepare_public_voice_samples.py" `
        --input-dir "$root\test_input" `
        --ffmpeg-binary "$root\bin\ffmpeg\ffmpeg.exe"
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Write-Host "Setup complete."
Write-Host "Input test folder:" "$root\test_input"
Write-Host "Output test folder:" "$root\test_output"
