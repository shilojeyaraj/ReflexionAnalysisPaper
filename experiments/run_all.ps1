# Run all 9 experiment conditions (3 backends x 3 domains) — Windows PowerShell
# Usage: from repo root,  pwsh -File experiments/run_all.ps1
$ErrorActionPreference = "Stop"

if (-not $env:OPENAI_API_KEY) {
    Write-Host "ERROR: OPENAI_API_KEY is not set." -ForegroundColor Red
    Write-Host 'Set it with: $env:OPENAI_API_KEY = "your_key"'
    exit 1
}

# Tool domain uses bundled bfcl_lite (no bfcl-eval required)
python -c "from environments.tool_env import ToolEnvironment; ToolEnvironment({})" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: ToolEnvironment import failed. Run from repo root after: pip install -e ." -ForegroundColor Yellow
}

$START = Get-Date
$RESULTS_DIR = ".\results"
New-Item -ItemType Directory -Force -Path $RESULTS_DIR | Out-Null

$BACKENDS = @("sliding_window", "sql", "vector")
$DOMAINS = @("code", "reasoning", "tool")

foreach ($backend in $BACKENDS) {
    foreach ($domain in $DOMAINS) {
        Write-Host ""
        Write-Host "=========================================="
        Write-Host "Starting: backend=$backend  domain=$domain"
        Write-Host "Timestamp: $(Get-Date -Format o)"
        Write-Host "=========================================="

        python experiments/run_experiment.py `
            --backend $backend `
            --domain $domain `
            --output-dir $RESULTS_DIR

        Write-Host "Completed: backend=$backend domain=$domain at $(Get-Date -Format o)"
        Start-Sleep -Seconds 5
    }
}

$ELAPSED = (Get-Date) - $START
Write-Host ""
Write-Host "=========================================="
Write-Host "All 9 conditions complete!"
Write-Host ("Total elapsed: {0}m {1}s" -f [int]$ELAPSED.TotalMinutes, $ELAPSED.Seconds)
Write-Host "Results saved to: $RESULTS_DIR"
Write-Host ""
Write-Host 'Plots: python -c "from analysis.plots import plot_all; plot_all(''./results'', ''./figures'')"'
Write-Host "=========================================="
