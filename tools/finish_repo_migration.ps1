# Complete the local directory switch after Codex releases its old workspace.
# Without -Finalize this script only verifies files. It never deletes the backup.
[CmdletBinding()]
param([switch]$Finalize)

$ErrorActionPreference = 'Stop'
$source = 'C:\Users\86153\Documents\Codex\2026-09-23\ai-jpeg-1-2-3-4'
$backup = 'C:\Users\86153\Documents\Codex\2026-09-23\ai-jpeg-1-2-3-4.migration-backup-2026-10-04'
$destination = 'E:\ai_image_origin_research\repo'
$auditPath = 'E:\ai_image_origin_research\cache\repository_migration_2026-10-04.sha256.json'

# Paths are deliberately fixed to this migration, not caller-supplied targets.
if ([IO.Path]::GetFullPath($PSScriptRoot) -ne "$destination\tools") {
    throw 'Run the verified script from the E: repository.'
}
if (-not (Test-Path -LiteralPath "$destination\.git\HEAD")) {
    throw 'The E: Git repository is missing.'
}
$sourceItem = Get-Item -LiteralPath $source -Force
if ($sourceItem.Attributes -band [IO.FileAttributes]::ReparsePoint) {
    if ($sourceItem.LinkType -eq 'Junction' -and @($sourceItem.Target)[0] -eq $destination) {
        Write-Output 'The C: workspace already points to the E: repository.'
        exit 0
    }
    throw 'The source is an unexpected directory link.'
}
if (Test-Path -LiteralPath $backup) {
    throw 'A backup already exists. Inspect it before retrying.'
}
try {
    $audit = Get-Content -LiteralPath $auditPath -Raw -Encoding UTF8 | ConvertFrom-Json
} catch {
    throw 'The UTF-8 migration audit could not be read. No directories were changed.'
}
if ($audit.Count -ne 8237) {
    throw 'The migration audit does not contain the expected 8237 files.'
}
$sourceEntries = @(Get-ChildItem -LiteralPath $source -Recurse -Force)
if (@($sourceEntries | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) {
    throw 'Unexpected directory links appeared in the old workspace.'
}
function Test-VolatileMigrationPath([string]$relative) {
    return ($relative -match '^\.git[\\/]' -or $relative -match '^\.venv[\\/]' -or
        $relative -match '(^|[\\/])(__pycache__|\.pytest_cache|\.ruff_cache|\.mypy_cache)[\\/]')
}
# Codex updates turn-diff Git refs while the task is open; package/test caches
# can also change. Business files, including local results, are checked strictly.
$stableAudit = @($audit | Where-Object { -not (Test-VolatileMigrationPath $_.RelativePath) })
$stableSourceFiles = @($sourceEntries | Where-Object {
    -not $_.PSIsContainer -and -not (Test-VolatileMigrationPath $_.FullName.Substring($source.Length + 1))
})
if ($stableSourceFiles.Count -ne $stableAudit.Count) {
    throw 'The old workspace file count changed. Preserve and review its changes.'
}

# Tracked sources may have been intentionally refactored after the copy. Their
# original bytes are preserved in Git history; only untracked local artifacts
# must still occupy their original paths on E:.
$originalCommit = '8a0e9c7d346b75e0391372ce890f213b3064705c'
$trackedPaths = @(& git -C $destination ls-tree -r --name-only $originalCommit)
if ($LASTEXITCODE -ne 0 -or $trackedPaths.Count -eq 0) {
    throw 'The original migration commit is missing from the E: Git history.'
}

foreach ($record in $stableAudit) {
    $relative = [string]$record.RelativePath
    $oldFile = Join-Path $source $relative
    $newFile = Join-Path $destination $relative
    if ((Get-Item -LiteralPath $oldFile).Length -ne $record.Bytes -or
        (Get-FileHash -LiteralPath $oldFile -Algorithm SHA256).Hash -ne $record.SHA256) {
        throw "The old workspace changed: $relative"
    }
    # Git, environment and disposable caches intentionally changed on E:.
    # Original tracked sources/reports are recoverable from the migration commit.
    if ($relative -match '^\.git[\\/]' -or $relative -match '^\.venv[\\/]' -or
        $relative -match '(^|[\\/])(__pycache__|\.pytest_cache|\.ruff_cache|\.mypy_cache)[\\/]' -or
        $trackedPaths -contains $relative.Replace('\', '/')) {
        continue
    }
    if (-not (Test-Path -LiteralPath $newFile -PathType Leaf) -or
        (Get-Item -LiteralPath $newFile).Length -ne $record.Bytes -or
        (Get-FileHash -LiteralPath $newFile -Algorithm SHA256).Hash -ne $record.SHA256) {
        throw "The destination differs from the migration audit: $relative"
    }
}
Write-Output 'Migration audit passed. Original business files remain unchanged; Git refs and caches are excluded.'
if (-not $Finalize) {
    Write-Output 'Verification only. Close Codex and run with -Finalize to switch directories.'
    exit 0
}

# Rename is reversible. If Windows still holds the directory open, it fails
# before any link is created. The copied E: repository is never modified here.
Move-Item -LiteralPath $source -Destination $backup
try {
    New-Item -ItemType Junction -Path $source -Target $destination | Out-Null
    $link = Get-Item -LiteralPath $source -Force
    if ($link.LinkType -ne 'Junction' -or @($link.Target)[0] -ne $destination) {
        throw 'The new directory link failed verification.'
    }
} catch {
    # Only remove the newly created junction, never its E: contents.
    if (Test-Path -LiteralPath $source) {
        $link = Get-Item -LiteralPath $source -Force
        if ($link.LinkType -eq 'Junction' -and @($link.Target)[0] -eq $destination) {
            Remove-Item -LiteralPath $source -Force
        } else {
            throw 'Unexpected source directory appeared; backup preserved for manual recovery.'
        }
    }
    Move-Item -LiteralPath $backup -Destination $source
    throw
}
Write-Output "C: workspace now points to $destination"
Write-Output "Old files preserved at $backup"
