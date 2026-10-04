<#
.SYNOPSIS
    Launch the occlubio surveillance console (FastAPI + SQLite + FAISS).

.DESCRIPTION
    Bootstraps the virtualenv if needed, verifies the inference + api extras are
    installed, then serves occlubio.api.app:app with uvicorn and opens the console
    in your browser. Ctrl+C stops the server.

.EXAMPLE
    .\run_server.ps1
    .\run_server.ps1 -Port 8080 -Reload
    .\run_server.ps1 -AuthorityCode "my-secret" -NoBrowser
    .\run_server.ps1 -Bind 0.0.0.0        # expose on the LAN (read the warning below)
    .\run_server.ps1 -Bind 0.0.0.0 -CertFile lan.pem -KeyFile lan-key.pem   # HTTPS for phones/iPad
#>
[CmdletBinding()]
param(
    [int]    $Port          = 8001,
    [string] $Bind          = "127.0.0.1",
    # Code authorities must supply when signing up. If empty, the server generates a
    # private code on first start (authority_code.txt) and prints it in the log.
    [string] $AuthorityCode = "",
    # sqlite:///occlubio.db unless overridden.
    [string] $DbUrl         = "",
    # TLS cert + key (e.g. from mkcert). Both set -> serve HTTPS, which iOS Safari
    # requires for webcam access from another device.
    [string] $CertFile      = "",
    [string] $KeyFile       = "",
    [switch] $Reload,
    [switch] $NoBrowser,
    # Skip the dependency probe when you know the venv is good (saves ~2s).
    [switch] $SkipDepCheck
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"

# --- venv -------------------------------------------------------------------
if (-not (Test-Path $py)) {
    Write-Host "[setup] no .venv found - creating one" -ForegroundColor Yellow

    $sysPy = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $sysPy) { throw "python not found on PATH. Install Python 3.10-3.12 first." }

    # insightface / onnxruntime / faiss have no wheels past 3.12 yet.
    $ver = & $sysPy -c "import sys; print('%d.%d' % sys.version_info[:2])"
    if ([version]$ver -lt [version]"3.10" -or [version]$ver -gt [version]"3.12") {
        Write-Host "[setup] WARNING: python $ver - the CV wheels only cover 3.10-3.12" -ForegroundColor Yellow
    }

    & $sysPy -m venv "$root\.venv"
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }

    & $py -m pip install --upgrade pip
    Write-Host "[setup] installing occlubio[infer,api] (first run pulls ~300MB of models later)" -ForegroundColor Yellow
    & $py -m pip install -e ".[infer,api]"
    if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }
}

# --- dependencies -----------------------------------------------------------
if (-not $SkipDepCheck) {
    & $py -c "import fastapi, uvicorn, sqlalchemy, insightface, onnxruntime, faiss, occlubio" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[setup] missing dependencies - installing occlubio[infer,api]" -ForegroundColor Yellow
        & $py -m pip install -e ".[infer,api]"
        if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }
    }
}

# --- port -------------------------------------------------------------------
$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    $owner = Get-Process -Id $busy[0].OwningProcess -ErrorAction SilentlyContinue
    $who = "pid $($busy[0].OwningProcess)"
    if ($owner) { $who = "$($owner.ProcessName) (pid $($owner.Id))" }
    throw "port $Port is already served by $who. Use -Port <n>, or stop it: Stop-Process -Id $($busy[0].OwningProcess)"
}

# --- environment ------------------------------------------------------------
if ($AuthorityCode) { $env:OCCLUBIO_AUTHORITY_CODE = $AuthorityCode }
if ($DbUrl)         { $env:OCCLUBIO_DB = $DbUrl }

if ($Bind -ne "127.0.0.1" -and $Bind -ne "localhost") {
    Write-Host "[warn] binding $Bind exposes a biometric system beyond this machine." -ForegroundColor Red
    Write-Host "[warn] See the responsible-use section of OCCLUSION_ROBUST_FR_ARCHITECTURE.md." -ForegroundColor Red
}

$scheme = "http"
if ($CertFile -or $KeyFile) {
    if (-not ($CertFile -and $KeyFile -and (Test-Path $CertFile) -and (Test-Path $KeyFile))) {
        throw "-CertFile and -KeyFile must both point to existing files"
    }
    $scheme = "https"
}

$url = "{2}://{0}:{1}/" -f $(if ($Bind -eq "0.0.0.0") { "127.0.0.1" } else { $Bind }), $Port, $scheme

# --- browser ----------------------------------------------------------------
# Uvicorn holds the foreground, so poll for readiness from a side job. Model load
# takes a few seconds on a cold start.
if (-not $NoBrowser) {
    Start-Job -ScriptBlock {
        param($u)
        for ($i = 0; $i -lt 90; $i++) {
            try {
                Invoke-WebRequest -Uri ($u + "login") -UseBasicParsing -TimeoutSec 3 | Out-Null
                Start-Process $u
                return
            } catch { Start-Sleep -Seconds 1 }
        }
    } -ArgumentList $url | Out-Null
}

# --- serve ------------------------------------------------------------------
Write-Host ""
Write-Host "  occlubio console  ->  $url" -ForegroundColor Green
Write-Host "  API docs          ->  ${url}docs" -ForegroundColor Green
Write-Host "  Ctrl+C to stop" -ForegroundColor DarkGray
Write-Host ""

$uvicornArgs = @("-m", "uvicorn", "occlubio.api.app:app", "--host", $Bind, "--port", "$Port")
if ($Reload) { $uvicornArgs += "--reload" }
if ($scheme -eq "https") { $uvicornArgs += @("--ssl-certfile", $CertFile, "--ssl-keyfile", $KeyFile) }

try {
    & $py @uvicornArgs
} finally {
    Get-Job | Where-Object { $_.State -eq "Running" } | Stop-Job -ErrorAction SilentlyContinue
    Get-Job | Remove-Job -Force -ErrorAction SilentlyContinue
    Write-Host "`n[occlubio] server stopped." -ForegroundColor DarkGray
}
