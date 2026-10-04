# Shared configuration for explicitly invoked historical PowerShell runners.
# Reading paths does not start a download or an inference process.
function Get-PalimpsestPaths {
    $repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
    $configPython = Join-Path $repo '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $configPython)) {
        throw 'Run uv sync --locked --extra dev in the repository first.'
    }
    $json = & $configPython -c 'import json; from dataclasses import asdict; from palimpsest.paths import PATHS; print(json.dumps({key: str(value) for key, value in asdict(PATHS).items()}))'
    if ($LASTEXITCODE -ne 0) {
        throw 'Palimpsest path configuration could not be loaded.'
    }
    return ($json | ConvertFrom-Json)
}
