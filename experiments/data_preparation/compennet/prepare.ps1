param()

$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
. (Join-Path $repoRoot 'tools\ResearchPaths.ps1')
$researchPaths = Get-PalimpsestPaths
$fileId = '1gUTWZLfiGRBgOnZe66ik-m315h4h0gPt'
$expectedBytes = [long]2282667301
$folder = Join-Path $researchPaths.data 'raw\compennet'
$finalPath = Join-Path $folder 'CompenNetDataset.zip'
$partialPath = "$finalPath.partial"
$landingPath = Join-Path $folder 'drive_landing.html'
$logPath = Join-Path $researchPaths.work 'compennet_download.log'

New-Item -ItemType Directory -Path $folder -Force | Out-Null
New-Item -ItemType Directory -Path $researchPaths.work -Force | Out-Null
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
        $landingUrl = "https://drive.google.com/uc?export=download&id=$fileId"
        $ErrorActionPreference = 'Continue'
        & curl.exe --location --connect-timeout 30 --max-time 60 --silent --show-error --output $landingPath $landingUrl 2>> $logPath
        $landingExit = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        if ($landingExit -ne 0) {
            Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) landing failed attempt=$attempt exit=$landingExit"
            Start-Sleep -Seconds 10
            continue
        }
        $html = Get-Content -LiteralPath $landingPath -Raw
        $match = [regex]::Match($html, 'name="uuid" value="([^"]+)"')
        if (-not $match.Success) { throw 'Google Drive did not offer a confirmation UUID.' }
        $uuid = $match.Groups[1].Value
        $downloadUrl = "https://drive.usercontent.google.com/download?id=$fileId&export=download&confirm=t&uuid=$uuid"
        $ErrorActionPreference = 'Continue'
        & curl.exe --location --continue-at - --connect-timeout 30 --speed-limit 1024 --speed-time 60 --silent --show-error --output $partialPath $downloadUrl 2>> $logPath
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
    & tar.exe -tf $partialPath > $null 2>> $logPath
    if ($LASTEXITCODE -ne 0) { throw 'ZIP directory traversal failed' }
    Move-Item -LiteralPath $partialPath -Destination $finalPath
    Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) size/list verified; full member CRC audit pending: $finalPath sha256=$sha"
} catch {
    Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) FAILED: $($_.Exception.Message)"
    exit 1
}
