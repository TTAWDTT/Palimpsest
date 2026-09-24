param()

$ErrorActionPreference = 'Stop'
$sourceUrl = 'https://dfd.inf.tu-dresden.de/dataset/HalftoneImages-Color.tar.gz'
$destinationDirectory = 'E:\ai_image_origin_research\data\raw\dfd_halftone_color'
$finalPath = Join-Path $destinationDirectory 'HalftoneImages-Color.tar.gz'
$partialPath = "$finalPath.partial"
$logPath = Join-Path $PSScriptRoot 'dfd_color_download.log'
$expectedBytes = [long]25818121374

New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null
try {
    if (Test-Path -LiteralPath $finalPath) {
        if ((Get-Item -LiteralPath $finalPath).Length -ne $expectedBytes) {
            throw 'Existing final archive has wrong length.'
        }
        Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) final archive already present"
        exit 0
    }
    for ($attempt = 1; $attempt -le 100; $attempt++) {
        $current = if (Test-Path -LiteralPath $partialPath) { (Get-Item -LiteralPath $partialPath).Length } else { 0 }
        if ($current -eq $expectedBytes) { break }
        if ($current -gt $expectedBytes) { throw "Partial file too large: $current" }
        $ErrorActionPreference = 'Continue'
        & curl.exe --location --continue-at - --connect-timeout 30 --speed-limit 1024 --speed-time 60 --silent --show-error --output $partialPath $sourceUrl 2>> $logPath
        $exitStatus = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        $current = (Get-Item -LiteralPath $partialPath).Length
        Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) attempt=$attempt exit=$exitStatus bytes=$current"
        if ($current -eq $expectedBytes) { break }
        Start-Sleep -Seconds 10
    }
    $current = (Get-Item -LiteralPath $partialPath).Length
    if ($current -ne $expectedBytes) { throw "Expected $expectedBytes bytes, got $current" }
    $shaAlgorithm = [System.Security.Cryptography.SHA256]::Create()
    $hashStream = [System.IO.File]::OpenRead($partialPath)
    try {
        $shaBytes = $shaAlgorithm.ComputeHash($hashStream)
    } finally {
        $hashStream.Dispose()
        $shaAlgorithm.Dispose()
    }
    $sha = [System.BitConverter]::ToString($shaBytes).Replace('-', '').ToLowerInvariant()
    Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) complete_size sha256=$sha; verifying tar/gzip"
    & tar.exe -tzf $partialPath > $null 2>> $logPath
    if ($LASTEXITCODE -ne 0) { throw 'tar/gzip stream verification failed' }
    Move-Item -LiteralPath $partialPath -Destination $finalPath
    Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) verified complete: $finalPath sha256=$sha"
} catch {
    Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) FAILED: $($_.Exception.Message)"
    exit 1
}
