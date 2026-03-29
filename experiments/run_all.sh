#!/usr/bin/env bash
set -e

# =============================================================================
# run_all.sh — Execute all 9 experiment conditions (3 backends × 3 domains)
# =============================================================================

# Check OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    echo "Run: export OPENAI_API_KEY=your_key"
    exit 1
fi

# Tool domain uses BFCL (Berkeley Function Calling Leaderboard) — no API key needed.
# Ensure bfcl-eval is installed (pip install bfcl-eval, requires Python >= 3.10)
if ! python -c "import bfcl_eval" 2>/dev/null; then
    echo "WARNING: bfcl-eval is not installed. Tool domain experiments will fail."
    echo "Run: pip install bfcl-eval  (requires Python >= 3.10)"
fi

START_TIME=$(date +%s)
RESULTS_DIR="./results"
mkdir -p "$RESULTS_DIR"

BACKENDS=("sliding_window" "sql" "vector")
DOMAINS=("code" "reasoning" "tool")

for backend in "${BACKENDS[@]}"; do
    for domain in "${DOMAINS[@]}"; do
        echo ""
        echo "=========================================="
        echo "Starting: backend=$backend  domain=$domain"
        echo "Timestamp: $(date)"
        echo "=========================================="

        python experiments/run_experiment.py \
            --backend "$backend" \
            --domain "$domain" \
            --output-dir "$RESULTS_DIR"

        echo "Completed: backend=$backend domain=$domain at $(date)"
        sleep 5
    done
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "All 9 conditions complete!"
echo "Total elapsed time: $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Generate analysis plots:"
echo "  python -c \"from analysis.plots import plot_all; plot_all('$RESULTS_DIR', './figures')\""
echo ""
echo "Generate LaTeX table:"
echo "  python -c \"from analysis.summary_table import build_summary_table, print_latex_table; print_latex_table(build_summary_table('$RESULTS_DIR'))\""
echo "=========================================="
