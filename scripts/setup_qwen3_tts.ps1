param(
    [string]$RuntimeRoot = "",
    [string]$Python = "",
    [ValidateSet("cpu", "cuda")]
    [string]$Torch = "cpu",
    [string]$QwenTTSVersion = "0.1.1",
    [string]$ModelId = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    [string]$TokenizerId = "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    [switch]$SkipModelDownload
)

$ErrorActionPreference = "Stop"

function Resolve-RuntimeRoot {
    if ($RuntimeRoot) { return $RuntimeRoot }
    if ($env:VOICE_BRIDGE_RUNTIME_ROOT) { return $env:VOICE_BRIDGE_RUNTIME_ROOT }
    if (Test-Path -LiteralPath "I:\") { return "I:\voice_bridge" }
    return (Join-Path (Get-Location).Path ".runtime")
}

function Invoke-Checked {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Command)
    Write-Host ""
    Write-Host ">>" ($Command -join " ")
    & $Command[0] @($Command | Select-Object -Skip 1)
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $($Command -join ' ')"
    }
}

$root = Resolve-RuntimeRoot
$basePython = if ($Python) {
    $Python
} elseif ($env:VOICE_BRIDGE_PYTHON) {
    $env:VOICE_BRIDGE_PYTHON
} elseif (Test-Path -LiteralPath "I:\Python311\python.exe") {
    "I:\Python311\python.exe"
} else {
    Join-Path $root "python311\python.exe"
}

if (-not (Test-Path -LiteralPath $basePython)) {
    throw "Base Python not found: $basePython"
}

$venv = Join-Path $root ".venv-qwen3-tts"
$qwenPython = Join-Path $venv "Scripts\python.exe"
$modelRoot = Join-Path $root "models"
$qwenRoot = Join-Path $modelRoot "qwen3-tts"
$modelDir = Join-Path $qwenRoot "0.6B-base"
$tokenizerDir = Join-Path $qwenRoot "tokenizer-12Hz"
$status = Join-Path $root "backend_status_qwen3_tts.json"

New-Item -ItemType Directory -Force -Path $qwenRoot | Out-Null
$env:PIP_CACHE_DIR = Join-Path $root "pip-cache"

if (-not (Test-Path -LiteralPath $qwenPython)) {
    Invoke-Checked $basePython -m venv $venv
}

Invoke-Checked $qwenPython -m pip install --upgrade "pip==26.1.2" "setuptools==81.0.0" "wheel==0.47.0"
Invoke-Checked $qwenPython -m pip install "qwen-tts==$QwenTTSVersion"

if (-not $SkipModelDownload) {
$code = @"
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$ModelId', local_dir=r'$modelDir', local_dir_use_symlinks=False)
snapshot_download(repo_id='$TokenizerId', local_dir=r'$tokenizerDir', local_dir_use_symlinks=False)
"@
    Write-Host ""
    Write-Host ">>" $qwenPython "-c <download Qwen3-TTS models>"
    & $qwenPython -c $code
    if ($LASTEXITCODE -ne 0) {
        throw "Qwen3-TTS model download failed with exit code ${LASTEXITCODE}"
    }
}

$report = [ordered]@{
    qwen_tts_python = $qwenPython
    qwen_tts_version = $QwenTTSVersion
    model_id = $ModelId
    model_dir = $modelDir
    model_exists = Test-Path -LiteralPath $modelDir
    tokenizer_id = $TokenizerId
    tokenizer_dir = $tokenizerDir
    tokenizer_exists = Test-Path -LiteralPath $tokenizerDir
}
$report | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $status -Encoding UTF8

Write-Host ""
Write-Host "Qwen3-TTS runtime ready"
Write-Host "Python: $qwenPython"
Write-Host "Model:  $modelDir"
Write-Host "Status: $status"
