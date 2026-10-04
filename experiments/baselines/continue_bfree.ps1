$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
. (Join-Path $repoRoot 'tools\ResearchPaths.ps1')
$researchPaths = Get-PalimpsestPaths
$env:PYTHONPATH = "$repoRoot\src;$repoRoot" + $(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" } else { '' })
$python = (Join-Path $researchPaths.envs 'bfree\Scripts\python.exe')
$datasetRoot = (Join-Path $researchPaths.data 'derived\rewind')
$manifest = (Join-Path $researchPaths.data 'manifests\rewind_archive_expected.csv')
$weightsRoot = (Join-Path $researchPaths.models 'bfree')
$vendorCode = (Join-Path $researchPaths.work 'vendor\bfree\code')
$previous = (Join-Path $researchPaths.work 'bfree_rewind_pass2.csv')

for ($pass = 3; $pass -le 7; $pass++) {
    $outputCsv = (Join-Path $researchPaths.work "bfree_rewind_pass$pass.csv")
    $summaryJson = (Join-Path $researchPaths.work "bfree_rewind_pass${pass}_summary.json")
    & $python -m experiments.baselines.run_bfree_baseline `
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
        Copy-Item -LiteralPath $outputCsv -Destination (Join-Path $researchPaths.work 'bfree_rewind_complete.csv') -Force
        Copy-Item -LiteralPath $summaryJson -Destination (Join-Path $researchPaths.work 'bfree_rewind_complete_summary.json') -Force
        Write-Output "Completed all $($summary.input_images) images in pass $pass"
        exit 0
    }
}

throw 'B-Free did not complete within five additional passes'
