param(
    [string]$PythonVersion = "3.11.9",
    [string]$RuntimeRoot = "",
    [string]$RuntimeDir = "python311",
    [switch]$SkipFFmpeg
)

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$root = (Get-Location).Path
if (-not $RuntimeRoot) {
    if ($env:VOICE_BRIDGE_RUNTIME_ROOT) {
        $RuntimeRoot = $env:VOICE_BRIDGE_RUNTIME_ROOT
    } elseif (Test-Path -LiteralPath "I:\") {
        $RuntimeRoot = "I:\voice_bridge"
    } else {
        $RuntimeRoot = Join-Path $root ".runtime"
    }
}
$runtimePath = Join-Path $RuntimeRoot $RuntimeDir
$downloadDir = Join-Path $RuntimeRoot "downloads"
$cacheDir = Join-Path $RuntimeRoot "pip-cache"
$ffmpegDir = Join-Path $RuntimeRoot "bin\ffmpeg"
$pythonZip = Join-Path $downloadDir "python-embed.zip"
$getPip = Join-Path $downloadDir "get-pip.py"
$ffmpegZip = Join-Path $downloadDir "ffmpeg.zip"

New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
New-Item -ItemType Directory -Force -Path $ffmpegDir | Out-Null

if (-not (Test-Path -LiteralPath $runtimePath)) {
    Invoke-WebRequest -Uri "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip" -OutFile $pythonZip
    Expand-Archive -LiteralPath $pythonZip -DestinationPath $runtimePath -Force
}

$pth = Get-ChildItem -LiteralPath $runtimePath -Filter "python*._pth" | Select-Object -First 1
if (-not $pth) {
    throw "Embedded Python ._pth file not found in $runtimePath"
}

$pthContent = Get-Content $pth.FullName
if (-not ($pthContent -contains "import site")) {
    $updated = $pthContent | ForEach-Object {
        if ($_ -eq "#import site") { "import site" } else { $_ }
    }
    Set-Content -LiteralPath $pth.FullName -Value $updated -Encoding ASCII
}

$python = Join-Path $runtimePath "python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "python.exe not found in $runtimePath"
}

if (-not (Test-Path -LiteralPath $getPip)) {
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $getPip
}

$env:PIP_CACHE_DIR = $cacheDir
& $python $getPip
& $python -m pip install --upgrade "pip==26.1.2" "setuptools==81.0.0" "wheel==0.47.0"
& $python -m pip install -r requirements.txt

if (-not $SkipFFmpeg -and -not (Test-Path -LiteralPath (Join-Path $ffmpegDir "ffmpeg.exe"))) {
    try {
        Invoke-WebRequest -Uri "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" -OutFile $ffmpegZip
        $extractDir = Join-Path $downloadDir "ffmpeg_extract"
        if (Test-Path -LiteralPath $extractDir) {
            Remove-Item -LiteralPath $extractDir -Recurse -Force
        }
        Expand-Archive -LiteralPath $ffmpegZip -DestinationPath $extractDir -Force
        $binDir = Get-ChildItem -LiteralPath $extractDir -Directory | Select-Object -First 1
        Copy-Item -LiteralPath (Join-Path $binDir.FullName "bin\ffmpeg.exe") -Destination (Join-Path $ffmpegDir "ffmpeg.exe") -Force
        Copy-Item -LiteralPath (Join-Path $binDir.FullName "bin\ffprobe.exe") -Destination (Join-Path $ffmpegDir "ffprobe.exe") -Force
    } catch {
        Write-Warning "ffmpeg download failed. Audio conversion tests may not work until ffmpeg is installed. $_"
    }
}

Write-Host "Python runtime:" $python
Write-Host "ffmpeg binary:" (Join-Path $ffmpegDir "ffmpeg.exe")
Write-Host "Runtime root:" $RuntimeRoot
