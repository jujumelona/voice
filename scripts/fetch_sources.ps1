param(
    [string]$Destination = "vendor",
    [ValidateSet("all", "fast", "balanced", "quality")]
    [string]$Mode = "all"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $Destination | Out-Null

function Clone-Or-Update {
    param(
        [string]$Name,
        [string]$Url
    )

    $target = Join-Path $Destination $Name
    if (Test-Path $target) {
        Write-Host "Updating $Name"
        git -C $target fetch --depth 1 origin
        git -C $target pull --ff-only
        return
    }

    Write-Host "Cloning $Name"
    git clone --depth 1 $Url $target
}

if ($Mode -eq "all" -or $Mode -eq "fast") {
    Clone-Or-Update "whisper.cpp" "https://github.com/ggml-org/whisper.cpp.git"
    Clone-Or-Update "argos-translate" "https://github.com/argosopentech/argos-translate.git"
}

if ($Mode -eq "all" -or $Mode -eq "balanced") {
    Clone-Or-Update "faster-whisper" "https://github.com/SYSTRAN/faster-whisper.git"
    Clone-Or-Update "argos-translate" "https://github.com/argosopentech/argos-translate.git"
}

if ($Mode -eq "all" -or $Mode -eq "quality") {
    Clone-Or-Update "faster-whisper" "https://github.com/SYSTRAN/faster-whisper.git"
    Clone-Or-Update "marian" "https://github.com/marian-nmt/marian.git"
}

if ($Mode -eq "all") {
    Clone-Or-Update "wespeaker" "https://github.com/wenet-e2e/wespeaker.git"
}

Write-Host "Done. Review licenses and model manifests before publishing binaries or weights."
