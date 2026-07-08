param(
    [string]$RuntimeRoot = "",
    [string]$Venv = "",
    [string]$Python = "",
    [string]$PythonVersion = "3.11",
    [string]$TorchVersion = "2.5.1",
    [ValidateSet("cpu", "cuda")]
    [string]$Torch = "cpu"
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
    return (Get-Location).Path
}

function Invoke-Checked {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Command
    )
    & $Command[0] @($Command | Select-Object -Skip 1)
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $($Command -join ' ')"
    }
}

function Resolve-PythonLauncher {
    if ($Python) {
        return @($Python)
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $requested = @("py", "-$PythonVersion")
        if (Test-PythonLauncher $requested) {
            return $requested
        }
        foreach ($version in @("3.11", "3.10")) {
            $candidate = @("py", "-$version")
            if (Test-PythonLauncher $candidate) {
                return $candidate
            }
        }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        if (Test-PythonLauncher @("python")) {
            return @("python")
        }
    }
    throw "Python 3.10 or 3.11 is required for WeSpeaker setup. Install Python 3.11 or pass -Python C:\Path\To\python.exe."
}

function Test-PythonLauncher {
    param([string[]]$Launcher)
    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Launcher[0] @($Launcher | Select-Object -Skip 1) -c "import sys; raise SystemExit(0 if sys.version_info[:2] in [(3, 10), (3, 11)] else 1)" *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $oldErrorAction
    }
}

$resolvedRoot = Resolve-RuntimeRoot
$venvPath = if ($Venv) { $Venv } else { Join-Path $resolvedRoot ".venv-speaker" }
$pipCache = Join-Path $resolvedRoot "pip-cache"
$modelRoot = if ($env:VOICE_BRIDGE_MODEL_ROOT) { $env:VOICE_BRIDGE_MODEL_ROOT } else { Join-Path $resolvedRoot "models" }
$wespeakerHome = if ($env:VOICE_BRIDGE_WESPEAKER_HOME) { $env:VOICE_BRIDGE_WESPEAKER_HOME } else { Join-Path $modelRoot "wespeaker" }
$pretrainDir = if ($env:VOICE_BRIDGE_WESPEAKER_ERES2NET_DIR) { $env:VOICE_BRIDGE_WESPEAKER_ERES2NET_DIR } else { Join-Path $modelRoot "speaker\wespeaker\eres2net-large" }

New-Item -ItemType Directory -Force -Path $resolvedRoot | Out-Null
New-Item -ItemType Directory -Force -Path $pipCache | Out-Null
New-Item -ItemType Directory -Force -Path $modelRoot | Out-Null
New-Item -ItemType Directory -Force -Path $wespeakerHome | Out-Null
$env:PIP_CACHE_DIR = $pipCache
$env:WESPEAKER_HOME = $wespeakerHome
$pythonLauncher = @(Resolve-PythonLauncher)

if (-not (Test-Path -LiteralPath $venvPath)) {
    Invoke-Checked @pythonLauncher -m venv $venvPath
}

$pythonExe = Join-Path $venvPath "Scripts\python.exe"
Invoke-Checked $pythonExe -m pip install -U pip setuptools wheel

if (-not (Test-Path -LiteralPath "vendor/wespeaker")) {
    Invoke-Checked git clone --depth 1 https://github.com/wenet-e2e/wespeaker.git vendor/wespeaker
}

if ($Torch -eq "cpu") {
    Invoke-Checked $pythonExe -m pip install --force-reinstall "torch==$TorchVersion+cpu" "torchaudio==$TorchVersion+cpu" --index-url https://download.pytorch.org/whl/cpu
} else {
    Invoke-Checked $pythonExe -m pip install --force-reinstall "torch==$TorchVersion" "torchaudio==$TorchVersion"
}

Invoke-Checked $pythonExe -m pip install -r requirements/speaker-encoder-wespeaker.txt

Write-Host "Speaker encoder runtime installed:" $pythonExe
Write-Host "Speaker encoder: WeSpeaker ERes2Net-large"
Write-Host "Python launcher:" ($pythonLauncher -join " ")
Write-Host "Runtime root:" $resolvedRoot
Write-Host "Pip cache:" $pipCache
Write-Host "WESPEAKER_HOME:" $wespeakerHome
Write-Host "Optional pretrain dir:" $pretrainDir
