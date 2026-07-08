param(
    [switch]$SkipSampleDownload,
    [switch]$SkipFFmpeg
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

if (-not $SkipSampleDownload) {
    $python = Join-Path $root ".runtime\python311\python.exe"
    & $python "$root\scripts\prepare_public_voice_samples.py" `
        --input-dir "$root\input_test_public_voice" `
        --ffmpeg-binary "$root\bin\ffmpeg\ffmpeg.exe"
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Write-Host "Setup complete."
Write-Host "Input test folder:" "$root\input_test_public_voice"
Write-Host "Output test folder:" "$root\output_test_public_voice"
