param(
    [string]$RuntimeRoot = "",
    [string]$Python = "",
    [string]$WhisperModel = "base",
    [string]$WhisperCppVersion = "v1.9.1",
    [string]$MarianVersion = "1.1.0",
    [string[]]$ArgosPair = @("en:ko", "ko:en"),
    [switch]$BuildMarian,
    [switch]$SkipWhisperBuild,
    [switch]$SkipWhisperModel,
    [switch]$SkipArgosPackages,
    [switch]$SkipMarianBuild,
    [switch]$SkipFFmpeg,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Resolve-RuntimeRoot {
    if ($RuntimeRoot) { return $RuntimeRoot }
    if ($env:VOICE_BRIDGE_RUNTIME_ROOT) { return $env:VOICE_BRIDGE_RUNTIME_ROOT }
    if (Test-Path -LiteralPath "I:\") { return "I:\voice_bridge" }
    return (Join-Path (Get-Location).Path ".runtime")
}

function Resolve-Python {
    if ($Python) { return $Python }
    if ($env:VOICE_BRIDGE_PYTHON) { return $env:VOICE_BRIDGE_PYTHON }
    if (Test-Path -LiteralPath "I:\Python311\python.exe") { return "I:\Python311\python.exe" }
    $candidate = Join-Path $script:Root "python311\python.exe"
    if (Test-Path -LiteralPath $candidate) { return $candidate }
    throw "Python not found. Run scripts/setup_all.ps1 first or pass -Python <python.exe>."
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

function Resolve-Tool {
    param([string[]]$Names)
    foreach ($name in $Names) {
        $found = Get-Command $name -ErrorAction SilentlyContinue
        if ($found) { return $found.Source }
    }
    return $null
}

function Invoke-DownloadFile {
    param(
        [string]$Url,
        [string]$OutFile
    )
    if (Test-Path -LiteralPath $OutFile) {
        Remove-Item -LiteralPath $OutFile -Force
    }
    $curl = Resolve-Tool @("curl.exe", "curl")
    if ($curl) {
        Invoke-Checked $curl --fail --location --retry 3 --retry-delay 2 --output $OutFile $Url
    } else {
        Invoke-WebRequest -Uri $Url -OutFile $OutFile
    }
    if ((Get-Item -LiteralPath $OutFile).Length -le 0) {
        throw "Download produced an empty file: $OutFile"
    }
}

function Install-SourceZip {
    param(
        [string]$Name,
        [string]$Url,
        [string]$Destination
    )
    if (Test-Path -LiteralPath $Destination) { return $Destination }

    $downloads = Join-Path $script:Root "downloads"
    $zip = Join-Path $downloads "$Name.zip"
    $extractDir = Join-Path $downloads "${Name}_extract"
    New-Item -ItemType Directory -Force -Path $downloads, (Split-Path $Destination) | Out-Null

    Invoke-DownloadFile -Url $Url -OutFile $zip
    if (Test-Path -LiteralPath $extractDir) {
        Remove-Item -LiteralPath $extractDir -Recurse -Force
    }
    Expand-Archive -LiteralPath $zip -DestinationPath $extractDir -Force
    $expanded = Get-ChildItem -LiteralPath $extractDir -Directory | Select-Object -First 1
    if (-not $expanded) { throw "Source archive did not contain a top-level directory: $zip" }
    if (Test-Path -LiteralPath $Destination) {
        Remove-Item -LiteralPath $Destination -Recurse -Force
    }
    Move-Item -LiteralPath $expanded.FullName -Destination $Destination
    return $Destination
}

function Test-WhisperSource {
    param([string]$Path)
    return (
        (Test-Path -LiteralPath (Join-Path $Path "CMakeLists.txt")) -and
        (Test-Path -LiteralPath (Join-Path $Path "examples\CMakeLists.txt")) -and
        (Test-Path -LiteralPath (Join-Path $Path "bindings\javascript\package-tmpl.json")) -and
        (Test-Path -LiteralPath (Join-Path $Path "models\download-ggml-model.cmd"))
    )
}

function Resolve-WhisperSource {
    $vendor = Join-Path (Get-Location).Path "vendor\whisper.cpp"
    if (Test-WhisperSource $vendor) { return $vendor }

    $source = Join-Path $script:Root "src\whisper.cpp-$WhisperCppVersion"
    if (Test-WhisperSource $source) { return $source }

    $url = "https://github.com/ggml-org/whisper.cpp/archive/refs/tags/$WhisperCppVersion.zip"
    $source = Install-SourceZip -Name "whisper.cpp-$WhisperCppVersion" -Url $url -Destination $source
    if (-not (Test-WhisperSource $source)) {
        throw "Downloaded whisper.cpp source is incomplete: $source"
    }
    return $source
}

function Test-MarianSource {
    param([string]$Path)
    $thirdPartyCmake = Join-Path $Path "src\3rd_party\CMakeLists.txt"
    $needsIntgemm = $false
    if (Test-Path -LiteralPath $thirdPartyCmake) {
        $needsIntgemm = (Select-String -LiteralPath $thirdPartyCmake -Pattern "intgemm" -Quiet)
    }
    $hasRequiredSubmodules = if ($needsIntgemm) {
        (Test-Path -LiteralPath (Join-Path $Path "src\3rd_party\intgemm\CMakeLists.txt")) -and
        (Test-Path -LiteralPath (Join-Path $Path "src\3rd_party\yaml-cpp\CMakeLists.txt"))
    } else {
        (Test-Path -LiteralPath (Join-Path $Path "src\3rd_party\yaml-cpp\CMakeLists.txt"))
    }
    return (
        (Test-Path -LiteralPath (Join-Path $Path "CMakeLists.txt")) -and
        (Test-Path -LiteralPath (Join-Path $Path "src\CMakeLists.txt")) -and
        (Test-Path -LiteralPath (Join-Path $Path "src\command\marian_decoder.cpp")) -and
        $hasRequiredSubmodules
    )
}

function Resolve-MarianSource {
    $vendor = Join-Path (Get-Location).Path "vendor\marian"
    if (Test-MarianSource $vendor) { return $vendor }

    $source = Join-Path $script:Root "src\marian-dev-$MarianVersion"
    if (Test-MarianSource $source) { return $source }

    $url = "https://github.com/marian-nmt/marian-dev/archive/refs/tags/$MarianVersion.zip"
    $source = Install-SourceZip -Name "marian-dev-$MarianVersion" -Url $url -Destination $source
    if (-not (Test-MarianSource $source)) {
        throw "Downloaded Marian source is incomplete: $source"
    }
    return $source
}

function Reset-BuildDir {
    param([string]$BuildDir)
    if (Test-Path -LiteralPath $BuildDir) {
        Remove-Item -LiteralPath $BuildDir -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
}

function Ensure-Ffmpeg {
    $ffmpegDir = Join-Path $script:Root "bin\ffmpeg"
    $ffmpegExe = Join-Path $ffmpegDir "ffmpeg.exe"
    if (Test-Path -LiteralPath $ffmpegExe) { return }

    $downloads = Join-Path $script:Root "downloads"
    $zip = Join-Path $downloads "ffmpeg.zip"
    $extractDir = Join-Path $downloads "ffmpeg_extract"
    New-Item -ItemType Directory -Force -Path $ffmpegDir, $downloads | Out-Null
    Invoke-DownloadFile -Url "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" -OutFile $zip
    if (Test-Path -LiteralPath $extractDir) {
        Remove-Item -LiteralPath $extractDir -Recurse -Force
    }
    Expand-Archive -LiteralPath $zip -DestinationPath $extractDir -Force
    $binDir = Get-ChildItem -LiteralPath $extractDir -Directory | Select-Object -First 1
    Copy-Item -LiteralPath (Join-Path $binDir.FullName "bin\ffmpeg.exe") -Destination $ffmpegExe -Force
    Copy-Item -LiteralPath (Join-Path $binDir.FullName "bin\ffprobe.exe") -Destination (Join-Path $ffmpegDir "ffprobe.exe") -Force
}

function Build-WhisperCpp {
    if ($SkipWhisperBuild) { return }
    $outExe = Join-Path $script:Root "bin\whisper-cli.exe"
    if ((Test-Path -LiteralPath $outExe) -and -not $Force) { return }

    $gcc = Resolve-Tool @("gcc.exe", "gcc")
    $gxx = Resolve-Tool @("g++.exe", "g++")
    $make = Resolve-Tool @("mingw32-make.exe", "mingw32-make")
    if (-not $gcc -or -not $gxx -or -not $make) {
        throw "MinGW gcc/g++/mingw32-make not found. Install MSYS2 MinGW64 or pass a PATH that contains them."
    }

    $source = Resolve-WhisperSource
    $build = Join-Path $script:Root "build\whisper.cpp"
    New-Item -ItemType Directory -Force -Path (Split-Path $outExe) | Out-Null
    Reset-BuildDir $build

    Invoke-Checked cmake `
        -S $source `
        -B $build `
        -G "MinGW Makefiles" `
        "-DCMAKE_MAKE_PROGRAM=$make" `
        "-DCMAKE_C_COMPILER=$gcc" `
        "-DCMAKE_CXX_COMPILER=$gxx" `
        -DCMAKE_BUILD_TYPE=Release `
        -DWHISPER_BUILD_TESTS=OFF `
        -DWHISPER_BUILD_SERVER=OFF `
        -DWHISPER_SDL2=OFF `
        -DWHISPER_COMMON_FFMPEG=OFF `
        -DBUILD_SHARED_LIBS=OFF
    Invoke-Checked cmake --build $build --config Release --target whisper-cli --parallel

    $built = Get-ChildItem -LiteralPath $build -Recurse -Filter "whisper-cli.exe" | Select-Object -First 1
    if (-not $built) { throw "whisper-cli.exe was not produced under $build" }
    Copy-Item -LiteralPath $built.FullName -Destination $outExe -Force
}

function Download-WhisperModel {
    if ($SkipWhisperModel) { return }
    $modelDir = Join-Path $script:ModelRoot "whisper"
    $modelPath = Join-Path $modelDir "ggml-$WhisperModel.bin"
    if ((Test-Path -LiteralPath $modelPath) -and -not $Force) { return }
    New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
    $source = Resolve-WhisperSource
    $downloadScript = Join-Path $source "models\download-ggml-model.cmd"
    Invoke-Checked cmd /c $downloadScript $WhisperModel $modelDir
}

function Install-ArgosPackages {
    if ($SkipArgosPackages) { return }
    $status = Join-Path $script:Root "backend_status_argos.json"
    $args = @("scripts\install_argos_language_packages.py", "--status-json", $status)
    foreach ($pair in $ArgosPair) {
        $args += @("--pair", $pair)
    }
    Invoke-Checked $script:PythonExe @args
}

function Build-Marian {
    if ($SkipMarianBuild) { return }
    if (-not $BuildMarian) { return }
    $outExe = Join-Path $script:Root "bin\marian-decoder.exe"
    if ((Test-Path -LiteralPath $outExe) -and -not $Force) { return }

    $gcc = Resolve-Tool @("gcc.exe", "gcc")
    $gxx = Resolve-Tool @("g++.exe", "g++")
    $make = Resolve-Tool @("mingw32-make.exe", "mingw32-make")
    if (-not $gcc -or -not $gxx -or -not $make) {
        throw "MinGW gcc/g++/mingw32-make not found. Install MSYS2 MinGW64 or pass a PATH that contains them."
    }

    $source = Resolve-MarianSource
    $build = Join-Path $script:Root "build\marian"
    New-Item -ItemType Directory -Force -Path (Split-Path $outExe) | Out-Null
    Reset-BuildDir $build

    Invoke-Checked cmake `
        -S $source `
        -B $build `
        -G "MinGW Makefiles" `
        "-DCMAKE_MAKE_PROGRAM=$make" `
        "-DCMAKE_C_COMPILER=$gcc" `
        "-DCMAKE_CXX_COMPILER=$gxx" `
        -DCMAKE_BUILD_TYPE=Release `
        -DCOMPILE_CUDA=OFF `
        -DCOMPILE_SERVER=OFF `
        -DCOMPILE_TESTS=OFF `
        -DCOMPILE_EXAMPLES=OFF `
        -DUSE_MKL=OFF `
        -DUSE_MPI=OFF `
        -DUSE_SENTENCEPIECE=OFF `
        -DUSE_STATIC_LIBS=OFF `
        -DBUILD_ARCH=x86-64
    Invoke-Checked cmake --build $build --config Release --target marian-decoder --parallel

    $built = Get-ChildItem -LiteralPath $build -Recurse -Filter "marian-decoder.exe" | Select-Object -First 1
    if (-not $built) { throw "marian-decoder.exe was not produced under $build" }
    Copy-Item -LiteralPath $built.FullName -Destination $outExe -Force
}

function Validate-Backends {
    $whisperExe = Join-Path $script:Root "bin\whisper-cli.exe"
    $marianExe = Join-Path $script:Root "bin\marian-decoder.exe"
    $ffmpegExe = Join-Path $script:Root "bin\ffmpeg\ffmpeg.exe"
    $whisperModel = Join-Path $script:ModelRoot "whisper\ggml-$WhisperModel.bin"

    $report = [ordered]@{
        python = $script:PythonExe
        runtime_root = $script:Root
        model_root = $script:ModelRoot
        ffmpeg = [ordered]@{ path = $ffmpegExe; exists = Test-Path -LiteralPath $ffmpegExe }
        whisper_cli = [ordered]@{ path = $whisperExe; exists = Test-Path -LiteralPath $whisperExe }
        whisper_model = [ordered]@{ path = $whisperModel; exists = Test-Path -LiteralPath $whisperModel }
        marian_decoder = [ordered]@{
            path = $marianExe
            exists = Test-Path -LiteralPath $marianExe
            requested = [bool]$BuildMarian
        }
        argos_status = Join-Path $script:Root "backend_status_argos.json"
    }

    if ($report.whisper_cli.exists) { Invoke-Checked $whisperExe --help }
    if ($report.marian_decoder.exists) { Invoke-Checked $marianExe --help }

    $statusPath = Join-Path $script:Root "backend_status.json"
    $report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statusPath -Encoding UTF8
    Write-Host ""
    Write-Host "Backend status:" $statusPath
    Get-Content -LiteralPath $statusPath
}

$script:Root = Resolve-RuntimeRoot
$script:PythonExe = Resolve-Python
$script:ModelRoot = if ($env:VOICE_BRIDGE_MODEL_ROOT) { $env:VOICE_BRIDGE_MODEL_ROOT } else { Join-Path $script:Root "models" }

New-Item -ItemType Directory -Force -Path $script:Root, $script:ModelRoot, (Join-Path $script:Root "bin") | Out-Null
$env:VOICE_BRIDGE_RUNTIME_ROOT = $script:Root
$env:VOICE_BRIDGE_MODEL_ROOT = $script:ModelRoot
$env:HF_HOME = Join-Path $script:ModelRoot "huggingface"
$env:PATH = (Join-Path $script:Root "bin") + ";" + (Join-Path $script:Root "bin\ffmpeg") + ";" + (Split-Path $script:PythonExe) + "\Scripts;" + $env:PATH

if (-not $SkipFFmpeg) {
    Ensure-Ffmpeg
}
Build-WhisperCpp
Download-WhisperModel
Install-ArgosPackages
Build-Marian
Validate-Backends
