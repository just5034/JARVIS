# Archive JARVIS models + DeepSeek V4 weights to external SSD.
#
# Usage:
#     pwsh scripts\archive_v4_to_ssd.ps1 -Phase 1
#     pwsh scripts\archive_v4_to_ssd.ps1 -Phase 2
#     pwsh scripts\archive_v4_to_ssd.ps1 -Phase 3
#     pwsh scripts\archive_v4_to_ssd.ps1 -Phase 4
#     pwsh scripts\archive_v4_to_ssd.ps1 -Phase All
#     pwsh scripts\archive_v4_to_ssd.ps1 -Verify
#     pwsh scripts\archive_v4_to_ssd.ps1 -Preflight
#
# Phases:
#   1 = Current models (Qwen3.5-27B + specialists + infra)        ~28 GB
#   2 = DeepSeek V4-Flash native FP4                              ~150 GB
#   3 = DeepSeek V4-Pro native FP4                                ~850 GB
#   4 = DeepSeek V4-Flash MLX 4-bit (optional, Apple Silicon)     ~145 GB

param(
    [ValidateSet("1","2","3","4","All")]
    [string]$Phase,
    [switch]$Verify,
    [switch]$Preflight,
    [string]$Root = "D:\jarvis-models"
)

$ErrorActionPreference = "Stop"
$script:FailedDownloads = @()

$LogDir = Join-Path $Root "logs"
$ChecksumFile = Join-Path $Root "checksums.txt"
$ManifestFile = Join-Path $Root "MANIFEST.md"

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] [$Level] $Message"
    Write-Host $line
    if (Test-Path $LogDir) {
        $logFile = Join-Path $LogDir ("archive_{0}.log" -f (Get-Date -Format "yyyyMMdd"))
        Add-Content -Path $logFile -Value $line
    }
}

function Invoke-Preflight {
    Write-Host "`n=== Preflight Checks ===`n"
    $pass = $true

    # 1. D: drive free space
    $d = Get-PSDrive D -ErrorAction SilentlyContinue
    if (-not $d) { Write-Host "  [FAIL] D: drive not found" -ForegroundColor Red; return }
    $freeGB = [math]::Round($d.Free / 1GB, 1)
    $needGB = if ($Phase -eq "3" -or $Phase -eq "All") { 1100 } elseif ($Phase -eq "2") { 200 } else { 50 }
    $ok = $freeGB -ge $needGB
    Write-Host ("  [{0}] D:\ free space: {1} GB (need {2} GB for phase {3})" -f $(if($ok){"PASS"}else{"FAIL"}), $freeGB, $needGB, $Phase) -ForegroundColor $(if($ok){"Green"}else{"Red"})
    if (-not $ok) { $pass = $false }

    # 2. Filesystem
    $vol = Get-Volume -DriveLetter D -ErrorAction SilentlyContinue
    $fs = if ($vol) { $vol.FileSystem } else { "unknown" }
    $ok = $fs -in @("NTFS","exFAT","ReFS")
    Write-Host ("  [{0}] D:\ filesystem: {1} (must be NTFS/exFAT, not FAT32)" -f $(if($ok){"PASS"}else{"FAIL"}), $fs) -ForegroundColor $(if($ok){"Green"}else{"Red"})
    if (-not $ok) { $pass = $false }

    # 3. Python
    try {
        $pyver = python -c "import sys; print(sys.version.split()[0])" 2>$null
        Write-Host "  [PASS] Python: $pyver" -ForegroundColor Green
    } catch {
        Write-Host "  [FAIL] Python not found on PATH" -ForegroundColor Red
        $pass = $false
    }

    # 4. huggingface_hub
    try {
        $hfver = python -c "import huggingface_hub; print(huggingface_hub.__version__)" 2>$null
        Write-Host "  [PASS] huggingface_hub: $hfver" -ForegroundColor Green
    } catch {
        Write-Host "  [FAIL] huggingface_hub not installed. Run: pip install huggingface_hub hf_transfer" -ForegroundColor Red
        $pass = $false
    }

    # 5. hf_transfer
    try {
        python -c "import hf_transfer" 2>$null
        Write-Host "  [PASS] hf_transfer installed (5x faster downloads)" -ForegroundColor Green
    } catch {
        Write-Host "  [WARN] hf_transfer not installed. Strongly recommended. Run: pip install hf_transfer" -ForegroundColor Yellow
    }

    # 6. HF auth
    try {
        $whoami = hf auth whoami 2>$null
        if ($whoami) {
            Write-Host "  [PASS] HF authenticated as: $whoami" -ForegroundColor Green
        } else {
            Write-Host "  [WARN] Not logged in to HF. DeepSeek repos are public but Qwen may gate. Run: hf auth login" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  [WARN] hf CLI not found or not logged in" -ForegroundColor Yellow
    }

    # 7. Network
    try {
        $null = Invoke-WebRequest -Uri "https://huggingface.co" -UseBasicParsing -TimeoutSec 10
        Write-Host "  [PASS] huggingface.co reachable" -ForegroundColor Green
    } catch {
        Write-Host "  [FAIL] Cannot reach huggingface.co" -ForegroundColor Red
        $pass = $false
    }

    Write-Host ""
    if ($pass) {
        Write-Host "All preflight checks passed.`n" -ForegroundColor Green
    } else {
        Write-Host "Preflight failed. Fix the above before running download phases.`n" -ForegroundColor Red
        exit 1
    }
}

function Initialize-Tree {
    $dirs = @(
        $Root,
        $LogDir,
        (Join-Path $Root "deepseek"),
        (Join-Path $Root "qwen"),
        (Join-Path $Root "specialists"),
        (Join-Path $Root "infrastructure"),
        (Join-Path $Root "adapters")
    )
    foreach ($d in $dirs) {
        if (-not (Test-Path $d)) {
            New-Item -ItemType Directory -Force -Path $d | Out-Null
        }
    }
}

function Set-DownloadEnv {
    $env:HF_HUB_ENABLE_HF_TRANSFER = "1"
    $env:HF_HOME = Join-Path $Root ".hf-cache"
    if (-not (Test-Path $env:HF_HOME)) {
        New-Item -ItemType Directory -Force -Path $env:HF_HOME | Out-Null
    }
    # Propagate stored HF token to subprocess env so child `hf download`
    # calls don't show "unauthenticated requests" warnings.
    $tokenFile = Join-Path $HOME ".cache\huggingface\token"
    if (Test-Path $tokenFile) {
        $env:HF_TOKEN = (Get-Content $tokenFile -Raw).Trim()
    }
}

function Get-Model {
    param(
        [string]$Repo,
        [string]$Dest,
        [string]$Label,
        [string]$SizeHint = "?"
    )
    Write-Log "[$Label] $Repo -> $Dest (approx $SizeHint)"
    if (-not (Test-Path $Dest)) {
        New-Item -ItemType Directory -Force -Path $Dest | Out-Null
    }
    $start = Get-Date
    hf download $Repo --local-dir $Dest
    $exit = $LASTEXITCODE
    $elapsed = (Get-Date) - $start
    if ($exit -ne 0) {
        Write-Log ("  [FAIL] $Label exited with code $exit after {0:N1} min" -f $elapsed.TotalMinutes) "ERROR"
        $script:FailedDownloads += "$Label ($Repo)"
        return
    }
    Write-Log ("  [DONE] $Label in {0:N1} min" -f $elapsed.TotalMinutes) "OK"
}

function Invoke-Phase1 {
    Write-Log "=== Phase 1: Current models mirror (~28 GB) ===" "PHASE"
    Get-Model "Qwen/Qwen3.5-27B"                       (Join-Path $Root "qwen\qwen3.5-27b-fp4")            "Qwen3.5-27B base"     "14 GB"
    Get-Model "Qwen/Qwen3.5-0.8B"                      (Join-Path $Root "infrastructure\qwen3.5-0.8b-draft") "Draft model"          "0.5 GB"
    Get-Model "launch/ThinkPRM-1.5B"                   (Join-Path $Root "infrastructure\think-prm-1.5b")    "ThinkPRM verifier"    "1 GB"
    Get-Model "sentence-transformers/all-MiniLM-L6-v2" (Join-Path $Root "infrastructure\all-minilm-l6-v2") "RAG embedding"        "0.1 GB"
    Get-Model "AI4Chem/ChemLLM-7B-Chat"                (Join-Path $Root "specialists\chemllm-7b")          "Chemistry specialist" "4 GB"
    Get-Model "BioMistral/BioMistral-7B"               (Join-Path $Root "specialists\biomistral-7b")       "Biology specialist"   "4 GB"
    Get-Model "EvolutionaryScale/esm3-sm-open-v1"      (Join-Path $Root "specialists\esm3-open")           "Protein specialist"   "1 GB"
    Get-Model "arcinstitute/evo2_7b"                   (Join-Path $Root "specialists\evo2-7b")             "Genomics specialist"  "4 GB"
    Write-Log "=== Phase 1 complete ===" "PHASE"
}

function Invoke-Phase2 {
    Write-Log "=== Phase 2: DeepSeek V4-Flash native FP4 (~150 GB) ===" "PHASE"
    Get-Model "deepseek-ai/DeepSeek-V4-Flash" (Join-Path $Root "deepseek\v4-flash-fp4") "V4-Flash FP4" "150 GB"
    Write-Log "=== Phase 2 complete ===" "PHASE"
}

function Invoke-Phase3 {
    Write-Log "=== Phase 3: DeepSeek V4-Pro native FP4 (~850 GB) ===" "PHASE"
    Write-Log "This will take 2-20 hours depending on bandwidth. Resumable on Ctrl-C." "WARN"
    Get-Model "deepseek-ai/DeepSeek-V4-Pro" (Join-Path $Root "deepseek\v4-pro-fp4") "V4-Pro FP4" "850 GB"
    Write-Log "=== Phase 3 complete ===" "PHASE"
}

function Invoke-Phase4 {
    Write-Log "=== Phase 4: DeepSeek V4-Flash MLX 4-bit (optional, ~145 GB) ===" "PHASE"
    Get-Model "mlx-community/DeepSeek-V4-Flash-4bit" (Join-Path $Root "deepseek\v4-flash-mlx-4bit") "V4-Flash MLX" "145 GB"
    Write-Log "=== Phase 4 complete ===" "PHASE"
}

function Invoke-Verify {
    Write-Log "=== Generating SHA256 checksums ===" "VERIFY"
    if (Test-Path $ChecksumFile) {
        $backup = $ChecksumFile + ".bak"
        Move-Item -Force $ChecksumFile $backup
        Write-Log "Existing checksums backed up to $backup"
    }
    $files = Get-ChildItem -Recurse -File $Root | Where-Object {
        $_.FullName -notlike "*\.hf-cache\*" -and
        $_.FullName -notlike "*\logs\*" -and
        $_.Name -ne "checksums.txt" -and
        $_.Name -ne "checksums.txt.bak"
    }
    $count = 0
    foreach ($f in $files) {
        $hash = (Get-FileHash $f.FullName -Algorithm SHA256).Hash
        $rel = $f.FullName.Substring($Root.Length).TrimStart('\')
        Add-Content -Path $ChecksumFile -Value "$hash  $rel"
        $count++
        if ($count % 50 -eq 0) { Write-Log "  hashed $count files..." }
    }
    Write-Log "Wrote $count checksums to $ChecksumFile" "OK"
}

function Write-Manifest {
    $totalGB = [math]::Round((Get-ChildItem -Recurse -File $Root | Measure-Object Length -Sum).Sum / 1GB, 1)
    $today = Get-Date -Format "yyyy-MM-dd"
    $content = @"
# JARVIS Model Archive — D:\jarvis-models

**Last updated:** $today
**Total size:** $totalGB GB
**Filesystem:** $(if($vol = Get-Volume -DriveLetter D -ErrorAction SilentlyContinue){$vol.FileSystem}else{"unknown"})

## Contents

| Path | Source | Size | Purpose |
|------|--------|------|---------|
| qwen/qwen3.5-27b-fp4 | Qwen/Qwen3.5-27B | ~14 GB | Active JARVIS base (offline mirror) |
| deepseek/v4-flash-fp4 | deepseek-ai/DeepSeek-V4-Flash | ~150 GB | Future hard-query escalation backend |
| deepseek/v4-pro-fp4 | deepseek-ai/DeepSeek-V4-Pro | ~850 GB | Archival; not runnable on current hardware |
| deepseek/v4-flash-mlx-4bit | mlx-community/DeepSeek-V4-Flash-4bit | ~145 GB | Mac Studio backup plan (optional) |
| specialists/* | various | ~13 GB | ChemLLM, BioMistral, ESM3, Evo 2 |
| infrastructure/* | various | ~1.6 GB | ThinkPRM, draft model, RAG embedder |
| adapters/* | local training output | (TBD) | HEP physics + HEP code LoRAs |

## Verification

Run \`pwsh scripts\archive_v4_to_ssd.ps1 -Verify\` to regenerate \`checksums.txt\`.

To compare with HF source, check the "Files and versions" tab on each repo and confirm
file count + total bytes. HF does not publish per-file SHA256, so we rely on hf-transfer's
on-the-fly integrity validation during download.

## How JARVIS uses this

\`configs/models.yaml\` has a \`deferred_backends\` section that points archive entries
at this directory. To activate one in production, change \`load_policy\` from \`archived\`
to \`on_demand\` (or \`always_resident\`) and ensure the path resolves in the deployment
environment (i.e., not D:\ — copy weights to the production model dir first).

## Why these models

- **V4-Flash** is the realistic escalation target. ~150 GB at native FP4 fits a Mac Studio M3 Ultra 256GB
  with KV cache headroom, or a multi-H100 box. Quality is comparable to Claude Sonnet 4.6 / GPT-5 Mini.
- **V4-Pro** is archival-only on consumer hardware. Even at native FP4 (~850 GB) it requires DGX Station
  GB300 (\$85K+) or an 8x H100 cluster. Worth keeping locally because DeepSeek has pulled weights before
  and MIT-licensed mirrors aren't guaranteed to persist.
- **Sub-Q4 quantization is NOT recommended for V4.** The model ships in native FP4+FP8 — there is no
  headroom left for 3-bit or 2-bit compression. Output breaks. Native FP4 IS the smallest safe form.
"@
    Set-Content -Path $ManifestFile -Value $content -Encoding UTF8
    Write-Log "Wrote manifest to $ManifestFile" "OK"
}

# --- main ---

if ($Preflight) {
    Invoke-Preflight
    exit 0
}

if ($Verify) {
    Initialize-Tree
    Invoke-Verify
    Write-Manifest
    exit 0
}

if (-not $Phase) {
    Write-Host @"
Usage:
  pwsh scripts\archive_v4_to_ssd.ps1 -Preflight
  pwsh scripts\archive_v4_to_ssd.ps1 -Phase 1     # ~28 GB, current models mirror
  pwsh scripts\archive_v4_to_ssd.ps1 -Phase 2     # ~150 GB, V4-Flash
  pwsh scripts\archive_v4_to_ssd.ps1 -Phase 3     # ~850 GB, V4-Pro (overnight)
  pwsh scripts\archive_v4_to_ssd.ps1 -Phase 4     # ~145 GB, V4-Flash MLX (optional)
  pwsh scripts\archive_v4_to_ssd.ps1 -Phase All
  pwsh scripts\archive_v4_to_ssd.ps1 -Verify      # regenerate checksums + manifest
"@
    exit 0
}

Initialize-Tree
Set-DownloadEnv
Invoke-Preflight

switch ($Phase) {
    "1"   { Invoke-Phase1 }
    "2"   { Invoke-Phase2 }
    "3"   { Invoke-Phase3 }
    "4"   { Invoke-Phase4 }
    "All" {
        Invoke-Phase1
        Invoke-Phase2
        Invoke-Phase3
        # Phase 4 intentionally not run by All; opt-in only
    }
}

Write-Manifest
if ($script:FailedDownloads.Count -gt 0) {
    Write-Log ("Phase $Phase finished with {0} failed download(s):" -f $script:FailedDownloads.Count) "WARN"
    foreach ($f in $script:FailedDownloads) { Write-Log "  - $f" "WARN" }
    Write-Log "Re-run the phase to retry, or fix repo paths in the script." "WARN"
} else {
    Write-Log "Phase $Phase done with no failures. Run -Verify to generate checksums when all phases complete." "OK"
}
