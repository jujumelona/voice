param(
    [string]$RuntimeRoot = "",
    [ValidateSet("cpu", "cuda")]
    [string]$Torch = "cpu",
    [string]$PythonVersion = "3.11.9",
    [string]$Python = "",
    [string]$WhisperModel = "base",
    [string[]]$ArgosPair = @("en:ko", "ko:en"),
    [switch]$BuildMarian,
    [switch]$SkipSpeaker,
    [switch]$SkipQwenTTS,
    [switch]$SkipQwenModelDownload,
    [switch]$SkipBackends,
    [switch]$SkipMarianBuild,
    [switch]$SkipFFmpeg,
    [switch]$SkipSmoke
)

$ErrorActionPreference = "Stop"

function Resolve-RuntimeRoot {
    if ($RuntimeRoot) {
        return $RuntimeRoot
    }
    if ($env:VOICE_BRIDGE_RUNTIME_ROOT) {
        return $env:VOICE_BRIDGE_RUNTIME_ROOT
    }
    if (Test-Path -LiteralPath "I:\") {
        return "I:\voice_bridge"
    }
    return (Join-Path (Get-Location).Path ".runtime")
}

function Invoke-Checked {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Command
    )
    Write-Host ""
    Write-Host ">>" ($Command -join " ")
    & $Command[0] @($Command | Select-Object -Skip 1)
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $($Command -join ' ')"
    }
}

$root = Resolve-RuntimeRoot
$python = if ($Python) {
    $Python
} elseif ($env:VOICE_BRIDGE_PYTHON) {
    $env:VOICE_BRIDGE_PYTHON
} elseif (Test-Path -LiteralPath "I:\Python311\python.exe") {
    "I:\Python311\python.exe"
} else {
    Join-Path $root "python311\python.exe"
}
$modelRoot = Join-Path $root "models"
$wespeakerHome = Join-Path $modelRoot "wespeaker"

New-Item -ItemType Directory -Force -Path $root | Out-Null
New-Item -ItemType Directory -Force -Path $modelRoot | Out-Null

$env:VOICE_BRIDGE_RUNTIME_ROOT = $root
$env:VOICE_BRIDGE_MODEL_ROOT = $modelRoot
$env:VOICE_BRIDGE_WESPEAKER_HOME = $wespeakerHome
$env:WESPEAKER_HOME = $wespeakerHome
$env:PIP_CACHE_DIR = Join-Path $root "pip-cache"

if (-not (Test-Path -LiteralPath $python)) {
    $setupRuntimeArgs = @(".\scripts\setup_runtime.ps1", "-RuntimeRoot", $root, "-RuntimeDir", "python311", "-PythonVersion", $PythonVersion)
    if ($SkipFFmpeg) {
        $setupRuntimeArgs += "-SkipFFmpeg"
    }
    Invoke-Checked powershell -ExecutionPolicy Bypass -File @setupRuntimeArgs
}

if (-not (Test-Path -LiteralPath $python)) {
    throw "Runtime Python was not created: $python"
}

Invoke-Checked $python -m pip install -r requirements.txt

if (-not $SkipSpeaker) {
    Invoke-Checked powershell -ExecutionPolicy Bypass -File .\scripts\setup_speaker_encoder.ps1 -RuntimeRoot $root -Python $python -Torch $Torch
    $env:VOICE_BRIDGE_WESPEAKER_PYTHON = Join-Path $root ".venv-speaker\Scripts\python.exe"
}

if (-not $SkipQwenTTS) {
    $qwenArgs = @(
        ".\scripts\setup_qwen3_tts.ps1",
        "-RuntimeRoot", $root,
        "-Python", $python,
        "-Torch", $Torch
    )
    if ($SkipQwenModelDownload) {
        $qwenArgs += "-SkipModelDownload"
    }
    Invoke-Checked powershell -ExecutionPolicy Bypass -File @qwenArgs
    $env:VOICE_BRIDGE_QWEN3_TTS_PYTHON = Join-Path $root ".venv-qwen3-tts\Scripts\python.exe"
}

if (-not $SkipBackends) {
    $backendArgs = @(
        ".\scripts\setup_backends.ps1",
        "-RuntimeRoot", $root,
        "-Python", $python,
        "-WhisperModel", $WhisperModel
    )
    if ($ArgosPair.Count -gt 0) {
        $backendArgs += @("-ArgosPair", ($ArgosPair -join ","))
    }
    if ($SkipFFmpeg) {
        $backendArgs += "-SkipFFmpeg"
    }
    if ($SkipMarianBuild) {
        $backendArgs += "-SkipMarianBuild"
    }
    if ($BuildMarian) {
        $backendArgs += "-BuildMarian"
    }
    Invoke-Checked powershell -ExecutionPolicy Bypass -File @backendArgs
}

if (-not $SkipSmoke) {
    Invoke-Checked $python scripts\check_runtime.py --strict
    Invoke-Checked $python -m voice_engine.pipeline.realtime_call_translate --validate-only --direction both --mode fast --voice-adapter spectral_delta
    Invoke-Checked $python scripts\check_publish_ready.py
}

Write-Host ""
Write-Host "Voice Engine setup complete"
Write-Host "Runtime root: $root"
Write-Host "Python: $python"
Write-Host "Model root: $modelRoot"
Write-Host "Check: $python scripts\check_runtime.py --strict"
Write-Host "Run:   $python -m voice_engine.pipeline.realtime_call_translate --direction both --mode fast --voice-adapter spectral_delta"
