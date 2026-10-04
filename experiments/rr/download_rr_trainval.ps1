$ErrorActionPreference = 'Stop'
$rawDirectory = 'E:\ai_image_origin_research\data\raw'
$logPath = 'E:\ai_image_origin_research\data\manifests\rr_trainval_download.log'
$partialPath = Join-Path $rawDirectory 'RRDataset_original_train_val.tar.gz.partial'
$finalPath = Join-Path $rawDirectory 'RRDataset_original_train_val.tar.gz'
$sourceUrl = 'https://zenodo.org/api/records/14963880/files/RRDataset_original_train_val.tar.gz/content'
$expectedLength = 2163176547
$expectedMd5 = '2f4498c3690d8f4c7a30d2e41dd34500'

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
    if (-not (Test-Path -LiteralPath $finalPath)) {
        for ($attempt = 1; $attempt -le 100; $attempt++) {
            $downloadedLength = (Get-Item -LiteralPath $partialPath).Length
            if ($downloadedLength -eq $expectedLength) { break }
            if ($downloadedLength -gt $expectedLength) { throw "Download exceeded expected length: $downloadedLength" }
            $ErrorActionPreference = 'Continue'
            & curl.exe --location --continue-at - --connect-timeout 30 --speed-limit 1024 --speed-time 60 --silent --show-error --output $partialPath $sourceUrl 2>> $logPath
            $exitCode = $LASTEXITCODE
            $ErrorActionPreference = 'Stop'
            $downloadedLength = (Get-Item -LiteralPath $partialPath).Length
            Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) attempt=$attempt exit=$exitCode bytes=$downloadedLength"
            if ($downloadedLength -eq $expectedLength) { break }
            Start-Sleep -Seconds 10
        }
        $downloadedLength = (Get-Item -LiteralPath $partialPath).Length
        if ($downloadedLength -ne $expectedLength) { throw "Expected $expectedLength bytes, got $downloadedLength" }
        $actualMd5 = Get-Md5Hex $partialPath
        if ($actualMd5 -ne $expectedMd5) { throw "MD5 mismatch: $actualMd5" }
        Move-Item -LiteralPath $partialPath -Destination $finalPath
    }
    Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) complete: $finalPath"
} catch {
    Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) failed: $($_.Exception.Message)"
    exit 1
}
