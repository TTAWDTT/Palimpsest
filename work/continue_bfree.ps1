$ErrorActionPreference = 'Stop'
$python = 'E:\ai_image_origin_research\envs\bfree\Scripts\python.exe'
$datasetRoot = 'E:\ai_image_origin_research\data\derived\rewind'
$manifest = 'E:\ai_image_origin_research\data\manifests\rewind_archive_expected.csv'
$weightsRoot = 'E:\ai_image_origin_research\models\bfree'
$vendorCode = 'work\vendor\bfree\code'
$previous = 'work\bfree_rewind_pass2.csv'

for ($pass = 3; $pass -le 7; $pass++) {
    $outputCsv = "work\bfree_rewind_pass$pass.csv"
    $summaryJson = "work\bfree_rewind_pass${pass}_summary.json"
    & $python 'work\run_bfree_baseline.py' `
        --vendor-code $vendorCode `
        --weights-root $weightsRoot `
        --dataset-root $datasetRoot `
        --manifest $manifest `
        --output-csv $outputCsv `
        --summary-json $summaryJson `
        --resume-from $previous `
        --tile-patches 64 `
        --max-new-images 800 `
        --warmup 20
    if ($LASTEXITCODE -ne 0) {
        throw "B-Free pass $pass exited with code $LASTEXITCODE"
    }
    $summary = Get-Content -LiteralPath $summaryJson -Raw | ConvertFrom-Json
    if ($summary.stopping_error) {
        throw "B-Free pass $pass stopped: $($summary.stopping_error)"
    }
    $previous = $outputCsv
    if ($summary.remaining_images -eq 0) {
        Copy-Item -LiteralPath $outputCsv -Destination 'work\bfree_rewind_complete.csv' -Force
        Copy-Item -LiteralPath $summaryJson -Destination 'work\bfree_rewind_complete_summary.json' -Force
        Write-Output "Completed all $($summary.input_images) images in pass $pass"
        exit 0
    }
}

throw 'B-Free did not complete within five additional passes'
