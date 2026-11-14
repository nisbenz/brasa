<#
.SYNOPSIS
    Downloads and unpacks the prebuilt harmonized dataset (~87 MB) for Brasa.

.DESCRIPTION
    PowerShell port of pipeline/fetch_prebuilt.sh for native Windows 10/11 support.
    Fetches the prebuilt data tables from GitHub releases (nisbenz/brasa) without
    requiring the full 9 GB FIRMS download, WSL, or Linux.
    Supports GitHub CLI (gh) if present, or falls back to Git Credential Manager / GitHub REST API.

.PARAMETER DataDir
    Target directory for unpacked data. Defaults to ~/firms/derived ($HOME\firms\derived),
    or the BRASA_DATA environment variable if set.

.EXAMPLE
    .\pipeline\fetch_prebuilt.ps1
    .\pipeline\fetch_prebuilt.ps1 "C:\Users\ADMIN\firms\derived"
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$DataDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Configuration
$Repo = "nisbenz/brasa"
$Tag = "data-2000-2025"
$AssetName = "brasa-data-2000-2025.tar.gz"
$ShaAssetName = "brasa-data-2000-2025.tar.gz.sha256"

# Resolve target data directory
if (-not $DataDir) {
    if ($env:BRASA_DATA) {
        $DataDir = $env:BRASA_DATA
    } else {
        $DataDir = Join-Path $HOME "firms\derived"
    }
}

# Expand tilde and normalize Windows path
if ($DataDir.StartsWith("~")) {
    $DataDir = Join-Path $HOME $DataDir.Substring(1).TrimStart('\', '/')
}
$DataDir = [System.IO.Path]::GetFullPath($DataDir)

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host " Brasa Prebuilt Data Fetcher (Windows PowerShell)" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Target Directory: $DataDir" -ForegroundColor Yellow

# Verify required system tools
$tarCmd = Get-Command tar.exe -ErrorAction SilentlyContinue
if (-not $tarCmd) {
    throw "tar.exe not found. Windows 10 (build 17063+) and Windows 11 include tar.exe by default in C:\Windows\System32."
}

$curlCmd = Get-Command curl.exe -ErrorAction SilentlyContinue

# Check if target data already exists
$expectedParquet = Join-Path $DataDir "app\harm.parquet"
if (Test-Path $expectedParquet) {
    Write-Host "`nPrebuilt data already exists in $DataDir" -ForegroundColor Green
    Write-Host "Found: $expectedParquet" -ForegroundColor Gray
    
    $defaultFirms = [System.IO.Path]::GetFullPath((Join-Path $HOME "firms\derived"))
    if ($DataDir.TrimEnd('\') -ne $defaultFirms.TrimEnd('\')) {
        Write-Host "`nTo run the server, set BRASA_DATA first:" -ForegroundColor Yellow
        Write-Host "  `$env:BRASA_DATA = '$DataDir'" -ForegroundColor White
    }
    Write-Host "Run the server from the 'app' directory:" -ForegroundColor Yellow
    Write-Host "  cd app" -ForegroundColor White
    Write-Host "  ..\.venv\Scripts\python -m uvicorn server:app --port 8765" -ForegroundColor White
    Write-Host "`nOpen your browser at: http://localhost:8765" -ForegroundColor Cyan
    return
}

# Persistent cache directory to avoid re-downloading if interrupted
$cacheDir = Join-Path ([System.IO.Path]::GetTempPath()) "brasa_data_cache"
New-Item -ItemType Directory -Path $cacheDir -Force | Out-Null

$cachedTar = Join-Path $cacheDir $AssetName
$cachedSha = Join-Path $cacheDir $ShaAssetName

# Temporary directory for extraction verification
$tempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("brasa_fetch_" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
Write-Host "Working cache directory: $cacheDir" -ForegroundColor Gray

try {
    $downloaded = $false

    # Method 1: GitHub CLI (gh)
    $ghCmd = Get-Command gh -ErrorAction SilentlyContinue
    if ($ghCmd) {
        Write-Host "`n--> Checking GitHub CLI (gh)..." -ForegroundColor Cyan
        $authCheck = & gh auth status 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "--> Downloading release assets via GitHub CLI..." -ForegroundColor Cyan
            & gh release download $Tag --repo $Repo --pattern "$AssetName*" --dir $cacheDir
            if ($LASTEXITCODE -eq 0) {
                $downloaded = $true
            }
        } else {
            Write-Host "GitHub CLI found but not logged in. Falling back to Git credentials / REST API..." -ForegroundColor Yellow
        }
    }

    # Method 2: Git Credential Manager / GitHub REST API
    if (-not $downloaded) {
        Write-Host "`n--> Checking download via GitHub REST API..." -ForegroundColor Cyan

        # Retrieve GitHub token from environment or Git Credential Manager
        $token = $env:GH_TOKEN
        if (-not $token) { $token = $env:GITHUB_TOKEN }
        if (-not $token) {
            $gitCmd = Get-Command git -ErrorAction SilentlyContinue
            if ($gitCmd) {
                try {
                    $credIn = "protocol=https`nhost=github.com`n"
                    $credOut = $credIn | & git credential fill 2>$null
                    foreach ($line in $credOut) {
                        if ($line -like "password=*") {
                            $token = $line.Substring(9).Trim()
                            break
                        }
                    }
                } catch {}
            }
        }

        # Query GitHub Release Metadata
        $headers = @{ "User-Agent" = "Brasa-Prebuilt-Fetcher" }
        if ($token) {
            $headers["Authorization"] = "Bearer $token"
        }

        $apiUrl = "https://api.github.com/repos/$Repo/releases/tags/$Tag"
        Write-Host "--> Querying release metadata for tag '$Tag'..." -ForegroundColor Gray
        
        $release = $null
        try {
            $release = Invoke-RestMethod -Uri $apiUrl -Headers $headers -Method Get
        } catch {
            Write-Warning "Could not fetch release metadata via API: $_"
        }

        if ($release -and $release.assets) {
            $assetObj = $release.assets | Where-Object { $_.name -eq $AssetName } | Select-Object -First 1
            $shaObj = $release.assets | Where-Object { $_.name -eq $ShaAssetName } | Select-Object -First 1

            if (-not $assetObj -or -not $shaObj) {
                throw "Could not locate required release assets ($AssetName and $ShaAssetName) in release $Tag."
            }

            # Download SHA256 checksum file
            Write-Host "--> Downloading checksum file ($ShaAssetName)..." -ForegroundColor Cyan
            if ($curlCmd) {
                & curl.exe -s -L -H "Authorization: Bearer $token" -H "Accept: application/octet-stream" -H "User-Agent: Brasa-Prebuilt-Fetcher" $shaObj.url -o $cachedSha
            } else {
                $shaHeaders = @{ "Accept" = "application/octet-stream"; "User-Agent" = "Brasa-Prebuilt-Fetcher" }
                if ($token) { $shaHeaders["Authorization"] = "Bearer $token" }
                Invoke-WebRequest -Uri $shaObj.url -Headers $shaHeaders -OutFile $cachedSha
            }

            # Check if previous partial exists in test_brasa_extract
            $existingPartial = Join-Path ([System.IO.Path]::GetTempPath()) "test_brasa_extract\brasa-data-2000-2025.tar.gz"
            if ((Test-Path $existingPartial) -and (-not (Test-Path $cachedTar))) {
                Copy-Item -Path $existingPartial -Destination $cachedTar -Force
            }

            # Download main tarball archive (~87 MB) with resume support
            Write-Host "--> Downloading dataset archive ($AssetName, ~87 MB)..." -ForegroundColor Cyan
            if ($curlCmd) {
                & curl.exe -C - -# -L -H "Authorization: Bearer $token" -H "Accept: application/octet-stream" -H "User-Agent: Brasa-Prebuilt-Fetcher" $assetObj.url -o $cachedTar
            } else {
                $tarHeaders = @{ "Accept" = "application/octet-stream"; "User-Agent" = "Brasa-Prebuilt-Fetcher" }
                if ($token) { $tarHeaders["Authorization"] = "Bearer $token" }
                Invoke-WebRequest -Uri $assetObj.url -Headers $tarHeaders -OutFile $cachedTar
            }

            if (Test-Path $cachedTar) {
                $downloaded = $true
            }
        } else {
            # Method 3: Direct Download fallback
            Write-Host "--> Trying direct release download..." -ForegroundColor Cyan
            $directTarUrl = "https://github.com/$Repo/releases/download/$Tag/$AssetName"
            $directShaUrl = "https://github.com/$Repo/releases/download/$Tag/$ShaAssetName"

            if ($curlCmd) {
                & curl.exe -s -L $directShaUrl -o $cachedSha
                & curl.exe -C - -# -L $directTarUrl -o $cachedTar
            } else {
                Invoke-WebRequest -Uri $directShaUrl -OutFile $cachedSha
                Invoke-WebRequest -Uri $directTarUrl -OutFile $cachedTar
            }
            if (Test-Path $cachedTar) {
                $downloaded = $true
            }
        }
    }

    if (-not $downloaded -or -not (Test-Path $cachedTar)) {
        throw "Failed to download prebuilt data. Ensure you have internet access and GitHub access to $Repo."
    }

    # Verify SHA-256 Checksum
    if (-not (Test-Path $cachedSha)) {
        throw "Checksum file was not found: $cachedSha"
    }

    Write-Host "`n--> Verifying SHA-256 checksum..." -ForegroundColor Cyan
    $expectedHash = ((Get-Content -Path $cachedSha -Raw).Trim() -split '\s+')[0].ToLower()
    $actualHash = (Get-FileHash -Path $cachedTar -Algorithm SHA256).Hash.ToLower()

    if ($actualHash -ne $expectedHash) {
        throw "Checksum verification failed!`nExpected: $expectedHash`nActual:   $actualHash"
    }
    Write-Host "--> Checksum verified: $actualHash" -ForegroundColor Green

    # Create target directory
    if (-not (Test-Path $DataDir)) {
        New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
        Write-Host "Created directory: $DataDir" -ForegroundColor Gray
    }

    # Unpack tar archive
    Write-Host "`n--> Unpacking data into $DataDir..." -ForegroundColor Cyan
    & tar.exe -xzf "$cachedTar" -C "$DataDir" --strip-components=1
    if ($LASTEXITCODE -ne 0) {
        throw "tar.exe failed with exit code $LASTEXITCODE."
    }

    # Verify expected unpacked files
    if (-not (Test-Path $expectedParquet)) {
        throw "Extraction completed, but expected file $expectedParquet was not found in $DataDir."
    }

    Write-Host "`n==================================================" -ForegroundColor Green
    Write-Host " Prebuilt data successfully unpacked into:" -ForegroundColor Green
    Write-Host "   $DataDir" -ForegroundColor White
    Write-Host "==================================================" -ForegroundColor Green

    $defaultFirms = [System.IO.Path]::GetFullPath((Join-Path $HOME "firms\derived"))
    if ($DataDir.TrimEnd('\') -ne $defaultFirms.TrimEnd('\')) {
        Write-Host "`nNOTE: You specified a custom data directory. Set BRASA_DATA before running the server:" -ForegroundColor Yellow
        Write-Host "  `$env:BRASA_DATA = '$DataDir'" -ForegroundColor White
    }

    Write-Host "`nTo start the Brasa server:" -ForegroundColor Cyan
    Write-Host "  cd app" -ForegroundColor White
    Write-Host "  ..\.venv\Scripts\python -m uvicorn server:app --port 8765" -ForegroundColor White
    Write-Host "`nLocalhost URL:" -ForegroundColor Green
    Write-Host "  http://localhost:8765" -ForegroundColor Yellow

} finally {
    # Clean up temporary working directory
    if (Test-Path $tempDir) {
        Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}
