param()

$ErrorActionPreference = 'Stop'
$sourceUrl = 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip'
$folder = 'E:\ai_image_origin_research\data\raw\div2k_scan'
$finalPath = Join-Path $folder 'DIV2K_valid_HR.zip'
$partialPath = "$finalPath.partial"
$logPath = Join-Path $PSScriptRoot 'div2k_valid_hr_download.log'
$expectedBytes = [long]448993893

New-Item -ItemType Directory -Path $folder -Force | Out-Null
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
    $sha = (Get-FileHash -LiteralPath $partialPath -Algorithm SHA256).Hash.ToLowerInvariant()
    & tar.exe -tf $partialPath > $null 2>> $logPath
    if ($LASTEXITCODE -ne 0) { throw 'ZIP traversal failed' }
    Move-Item -LiteralPath $partialPath -Destination $finalPath
    Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) size/list verified; member CRC audit pending: $finalPath sha256=$sha"
} catch {
    Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) FAILED: $($_.Exception.Message)"
    exit 1
}
