param(
    [Parameter(Mandatory = $true)][string]$SourceUrl,
    [Parameter(Mandatory = $true)][string]$FinalPath,
    [Parameter(Mandatory = $true)][long]$ExpectedLength,
    [Parameter(Mandatory = $true)][string]$ExpectedMd5,
    [Parameter(Mandatory = $true)][string]$LogPath
)

$ErrorActionPreference = 'Stop'
$partialPath = "$FinalPath.partial"

function Get-Md5Hex([string]$Path) {
    $stream = [System.IO.File]::OpenRead($Path)
    $hashAlgorithm = [System.Security.Cryptography.MD5]::Create()
    try {
        $hashBytes = $hashAlgorithm.ComputeHash($stream)
        return [System.BitConverter]::ToString($hashBytes).Replace('-', '').ToLowerInvariant()
    } finally {
        $hashAlgorithm.Dispose()
        $stream.Dispose()
    }
}

try {
    if (Test-Path -LiteralPath $FinalPath) {
        $existingLength = (Get-Item -LiteralPath $FinalPath).Length
        $existingMd5 = Get-Md5Hex $FinalPath
        if ($existingLength -ne $ExpectedLength -or $existingMd5 -ne $ExpectedMd5) {
            throw "Existing final file does not match official size and MD5: $FinalPath"
        }
        Add-Content -LiteralPath $LogPath -Value "$(Get-Date -Format o) already verified: $FinalPath"
        exit 0
    }

    for ($attempt = 1; $attempt -le 100; $attempt++) {
        $downloadedLength = if (Test-Path -LiteralPath $partialPath) {
            (Get-Item -LiteralPath $partialPath).Length
        } else { 0 }
        if ($downloadedLength -eq $ExpectedLength) { break }
        if ($downloadedLength -gt $ExpectedLength) {
            throw "Download exceeded expected length: $downloadedLength"
        }

        $ErrorActionPreference = 'Continue'
        & curl.exe --location --continue-at - --connect-timeout 30 --speed-limit 1024 --speed-time 60 --silent --show-error --output $partialPath $SourceUrl 2>> $LogPath
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        $downloadedLength = (Get-Item -LiteralPath $partialPath).Length
        Add-Content -LiteralPath $LogPath -Value "$(Get-Date -Format o) attempt=$attempt exit=$exitCode bytes=$downloadedLength"
        if ($downloadedLength -eq $ExpectedLength) { break }
        Start-Sleep -Seconds 10
    }

    $downloadedLength = (Get-Item -LiteralPath $partialPath).Length
    if ($downloadedLength -ne $ExpectedLength) {
        throw "Expected $ExpectedLength bytes, got $downloadedLength"
    }
    $actualMd5 = Get-Md5Hex $partialPath
    if ($actualMd5 -ne $ExpectedMd5.ToLowerInvariant()) {
        throw "MD5 mismatch: $actualMd5"
    }
    Move-Item -LiteralPath $partialPath -Destination $FinalPath
    Add-Content -LiteralPath $LogPath -Value "$(Get-Date -Format o) complete: $FinalPath"
} catch {
    Add-Content -LiteralPath $LogPath -Value "$(Get-Date -Format o) failed: $($_.Exception.Message)"
    exit 1
}
