$ErrorActionPreference = 'Stop'
$python = 'E:\ai_image_origin_research\envs\bfree\Scripts\python.exe'
$manifest = 'E:\ai_image_origin_research\data\manifests\rr_test_bfree_manifest.csv'
$datasetRoot = 'E:\ai_image_origin_research\data\derived\rr_test'
$weightsRoot = 'E:\ai_image_origin_research\models\bfree'
$auditPath = 'work\rr_test_audit.json'
$archiveManifest = 'E:\ai_image_origin_research\data\manifests\rr_test_archive_files.csv'
$expectedImages = 50999

function Get-CsvDataRowCount([string]$path) {
    if (-not (Test-Path -LiteralPath $path)) { return -1 }
    $lineCount = 0
    foreach ($line in [System.IO.File]::ReadLines((Resolve-Path -LiteralPath $path).Path)) {
        $lineCount++
    }
    return ($lineCount - 1)
}

if (-not (Test-Path -LiteralPath $manifest) -or -not (Test-Path -LiteralPath $datasetRoot)) {
    throw 'RRDataset test audit has not produced final files'
}
$audit = Get-Content -LiteralPath $auditPath -Raw | ConvertFrom-Json
if ($audit.decode_error_count -ne 0 -or $audit.exact_trainval_overlap_count -ne 14 -or
    -not $audit.accepted_14_exact_trainval_overlaps -or
    $audit.source_id_trainval_overlap_count -ne 0 -or $audit.decoded_images -ne 50999 -or
    $audit.duplicate_condition_source_groups -ne 0 -or -not $audit.audit_passed -or
    -not $audit.accepted_one_missing_exception -or
    (Get-CsvDataRowCount $manifest) -ne $expectedImages -or
    (Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash.ToLowerInvariant() -ne $audit.bfree_manifest_sha256 -or
    (Get-FileHash -LiteralPath $archiveManifest -Algorithm SHA256).Hash.ToLowerInvariant() -ne $audit.archive_manifest_sha256) {
    throw 'RRDataset test audit did not pass the declared one-missing-image gate'
}
$fingerprintFiles = @(
    $manifest,
    'experiments\baselines\run_bfree_baseline.py',
    "$weightsRoot\BFREE_dino2reg4\config.yaml",
    "$weightsRoot\BFREE_dino2reg4\model_epoch_best.pth"
)
$fingerprintFiles += @(Get-ChildItem -LiteralPath 'work\vendor\bfree\code' -Recurse -File -Filter '*.py' |
    Sort-Object FullName | ForEach-Object { $_.FullName })
$inputFingerprint = ($fingerprintFiles | ForEach-Object {
    (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()
}) -join ':'

$previous = $null
$previousCompleted = 0
for ($pass = 1; $pass -le 100; $pass++) {
    $outputCsv = "work\rr_bfree_pass_$pass.csv"
    $summaryJson = "work\rr_bfree_pass_${pass}_summary.json"

    $expectedCompleted = [Math]::Min($expectedImages, $previousCompleted + 1000)
    $passValid = $false
    if ((Test-Path -LiteralPath $outputCsv) -and
        (Test-Path -LiteralPath $summaryJson)) {
        try {
            $existingSummary = Get-Content -LiteralPath $summaryJson -Raw | ConvertFrom-Json
            $passValid = (
                $existingSummary.input_images -eq $expectedImages -and
                $existingSummary.completed_images -eq $expectedCompleted -and
                $existingSummary.remaining_images -eq ($expectedImages - $expectedCompleted) -and
                $existingSummary.errors -eq 0 -and -not $existingSummary.stopping_error -and
                $existingSummary.input_fingerprint -eq $inputFingerprint -and
                $existingSummary.output_csv_sha256 -eq (Get-FileHash -LiteralPath $outputCsv -Algorithm SHA256).Hash.ToLowerInvariant() -and
                (Get-CsvDataRowCount $outputCsv) -eq $expectedCompleted
            )
        } catch { $passValid = $false }
    }
    if (-not $passValid) {
        if ((Test-Path -LiteralPath $outputCsv) -or (Test-Path -LiteralPath $summaryJson)) {
            throw "Existing pass $pass does not match the current code/manifest fingerprint. Restore the recorded code version or use a separate output location; existing results were not changed."
        }
        $arguments = @(
            '-m', 'experiments.baselines.run_bfree_baseline',
            '--vendor-code', 'work\vendor\bfree\code',
            '--weights-root', $weightsRoot,
            '--dataset-root', $datasetRoot,
            '--manifest', $manifest,
            '--output-csv', $outputCsv,
            '--summary-json', $summaryJson,
            '--tile-patches', '64',
            '--max-new-images', '1000',
            '--warmup', '20'
        )
        if ($previous) {
            $arguments += @('--resume-from', $previous)
        }
        & $python @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "B-Free RR pass $pass exited with code $LASTEXITCODE"
        }
    }

    if (-not (Test-Path -LiteralPath $summaryJson)) {
        throw "B-Free RR pass $pass has no successful summary"
    }
    $summary = Get-Content -LiteralPath $summaryJson -Raw | ConvertFrom-Json
    if ($summary.stopping_error -or $summary.errors -ne 0 -or
        $summary.input_images -ne $expectedImages -or
        $summary.completed_images -ne $expectedCompleted -or
        $summary.remaining_images -ne ($expectedImages - $expectedCompleted) -or
        (Get-CsvDataRowCount $outputCsv) -ne $expectedCompleted) {
        throw "B-Free RR pass $pass stopped: $($summary.stopping_error)"
    }
    if (-not $passValid) {
        $summary | Add-Member -NotePropertyName input_fingerprint -NotePropertyValue $inputFingerprint
        $summary | Add-Member -NotePropertyName output_csv_sha256 -NotePropertyValue (
            (Get-FileHash -LiteralPath $outputCsv -Algorithm SHA256).Hash.ToLowerInvariant()
        )
        $temporarySummary = "$summaryJson.partial"
        $summary | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $temporarySummary -Encoding UTF8
        Move-Item -LiteralPath $temporarySummary -Destination $summaryJson -Force
    }
    $previous = $outputCsv
    $previousCompleted = $expectedCompleted
    Write-Output "pass=$pass completed=$($summary.completed_images) remaining=$($summary.remaining_images)"
    if ($summary.remaining_images -eq 0) {
        if ($summary.completed_images -ne $expectedImages) { throw 'Incomplete RR output' }
        Copy-Item -LiteralPath $outputCsv -Destination 'work\rr_bfree_complete.csv' -Force
        Copy-Item -LiteralPath $summaryJson -Destination 'work\rr_bfree_complete_summary.json' -Force
        Write-Output "COMPLETE images=$($summary.completed_images)"
        exit 0
    }
}

throw 'B-Free RR test did not complete within 100 passes'
