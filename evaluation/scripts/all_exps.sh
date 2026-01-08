#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EVAL_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT"

MANIFEST="$EVAL_DIR/data_manifests/tmp_datasets_rows2048.txt"
RAW_DIR="$REPO_ROOT/artifacts/reports/raw"
SUM_DIR="$REPO_ROOT/artifacts/reports/summaries"
PLOTS_DIR="$REPO_ROOT/artifacts/plots_png"
LOG_DIR="$REPO_ROOT/artifacts/logs"
mkdir -p "$RAW_DIR" "$SUM_DIR" "$PLOTS_DIR" "$LOG_DIR"

DATASETS="$(tr '\n' ' ' < "$MANIFEST")"

ROWCAPS_FULL=(32 64 128 256 512 1024 2048)
ROWCAPS_ICL=(32 64 128)
SPLITS=(random ood)

run_generic() {
  local model="$1"
  local split="$2"
  local rows="$3"
  local split_flag="--ood_split"
  [[ "$split" == "random" ]] && split_flag="--random_split"
  local out="$RAW_DIR/${model}_rows${rows}_standardized_report_${split}.json"
  echo "[generic] model=${model} split=${split} rows=${rows}"
  python3 "$EVAL_DIR/scripts/eval_generic_regression.py" \
    --model "$model" \
    --standardize \
    "$split_flag" \
    --max_rows "$rows" \
    --report_json "$out" \
    $DATASETS
}

run_xgboost() {
  local split="$1"; local rows="$2"
  local split_flag="--ood_split"
  [[ "$split" == "random" ]] && split_flag="--random_split"
  local out="$RAW_DIR/xgboost_rows${rows}_standardized_report_${split}.json"
  echo "[xgboost] split=${split} rows=${rows}"
  python3 "$EVAL_DIR/scripts/eval_xgboost_regression.py" \
    --standardize \
    "$split_flag" \
    --max_rows "$rows" \
    --report_json "$out" \
    $DATASETS
}

run_tabpfn_v25() {
  local split="$1"; local rows="$2"
  local split_flag="--ood_split"
  [[ "$split" == "random" ]] && split_flag="--random_split"
  local out="$RAW_DIR/tabpfn25_rows${rows}_standardized_report_${split}.json"
  local log="$LOG_DIR/tabpfn25_rate_limit.log"
  echo "[tabpfn v2.5] split=${split} rows=${rows}"
  if python3 "$EVAL_DIR/scripts/eval_tabpfn_client.py" \
      --manifest "$MANIFEST" \
      --rowcap "$rows" \
      "$split_flag" \
      --report_json "$out"; then
    echo "[tabpfn2.5] completed rows=${rows} split=${split}"
  else
    echo "[tabpfn2.5] rate-limit rows=${rows} split=${split}" | tee -a "$log"
  fi
}

run_icl() {
  local split="$1"; local rows="$2"
  local split_flag="--ood_split"
  [[ "$split" == "random" ]] && split_flag="--random_split"
  local out="$RAW_DIR/icl_llm_rows${rows}_${split}.json"
  local log="$RAW_DIR/icl_llm_rows${rows}_${split}_log.jsonl"
  echo "[icl] split=${split} rows=${rows}"
  python3 "$EVAL_DIR/scripts/eval_icl_llm.py" \
    --manifest "$MANIFEST" \
    --rowcap "$rows" \
    "$split_flag" \
    --out "$out" \
    --log_jsonl "$log"
}

main() {
  local generic_models=(random_forest lightgbm catboost realmlp tabm xrfm)
  for split in "${SPLITS[@]}"; do
    for rows in "${ROWCAPS_FULL[@]}"; do
      run_tabpfn_v25 "$split" "$rows"
      run_xgboost "$split" "$rows"
      for model in "${generic_models[@]}"; do
        run_generic "$model" "$split" "$rows"
      done
    done
    for rows in "${ROWCAPS_ICL[@]}"; do
      run_icl "$split" "$rows"
    done
  done

  python3 "$EVAL_DIR/scripts/aggregate_rowcap_reports.py" \
    --metrics_out "$SUM_DIR/model_rowcap_metrics.tsv" \
    --summary_out "$SUM_DIR/model_rowcap_summary.tsv"

  python3 "$EVAL_DIR/scripts/plot_consistency_curves.py"
}

main "$@"
