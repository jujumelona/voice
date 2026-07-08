param(
    [string]$RuntimeRoot = "",
    [string]$Python = "",
    [ValidateSet("outbound", "inbound", "both")]
    [string]$Direction = "both",
    [string]$SourceLanguage = "en",
    [string]$TargetLanguage = "ko",
    [string]$InputDevice = "",
    [string]$OutputDevice = "",
    [switch]$ValidateOnly,
    [switch]$ListAudioDevices,
    [ValidateSet("none", "spectral_delta")]
    [string]$VoiceAdapter = "spectral_delta"
)

$ErrorActionPreference = "Stop"

function Resolve-RuntimeRoot {
    if ($RuntimeRoot) { return $RuntimeRoot }
    if ($env:VOICE_BRIDGE_RUNTIME_ROOT) { return $env:VOICE_BRIDGE_RUNTIME_ROOT }
    if (Test-Path -LiteralPath "I:\") { return "I:\voice_bridge" }
    return (Join-Path (Get-Location).Path ".runtime")
}

$root = Resolve-RuntimeRoot
$pythonExe = if ($Python) {
    $Python
} elseif ($env:VOICE_BRIDGE_PYTHON) {
    $env:VOICE_BRIDGE_PYTHON
} elseif (Test-Path -LiteralPath "I:\Python311\python.exe") {
    "I:\Python311\python.exe"
} else {
    Join-Path $root "python311\python.exe"
}

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Runtime Python not found: $pythonExe. Run .\scripts\setup_all.ps1 first."
}

$env:VOICE_BRIDGE_RUNTIME_ROOT = $root
$env:VOICE_BRIDGE_MODEL_ROOT = Join-Path $root "models"
$env:VOICE_BRIDGE_QWEN3_TTS_PYTHON = Join-Path $root ".venv-qwen3-tts\Scripts\python.exe"

$argsList = @(
    "-m", "voice_engine.pipeline.realtime_call_translate",
    "--direction", $Direction,
    "--mode", "fast",
    "--decoder", "qwen3-tts",
    "--device", "cpu",
    "--voice-adapter", $VoiceAdapter,
    "--outbound-source-language", $SourceLanguage,
    "--outbound-target-language", $TargetLanguage,
    "--inbound-source-language", $TargetLanguage,
    "--inbound-target-language", $SourceLanguage
)

if ($InputDevice) {
    $argsList += @("--input-device", $InputDevice)
}
if ($OutputDevice) {
    $argsList += @("--output-device", $OutputDevice)
}
if ($ValidateOnly) {
    $argsList += "--validate-only"
}
if ($ListAudioDevices) {
    $argsList = @("-m", "voice_engine.pipeline.realtime_call_translate", "--list-audio-devices")
}

Write-Host ">> $pythonExe $($argsList -join ' ')"
& $pythonExe @argsList
exit $LASTEXITCODE
