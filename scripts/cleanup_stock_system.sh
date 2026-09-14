#!/usr/bin/env bash
# =============================================================================
# 股票系统定期清理规则（与 OpenClaw Cron 每日 02:00 任务一致）
# -----------------------------------------------------------------------------
# 目录基准：STOCK_SYSTEM_ROOT，未设置则为脚本所在 stock_system 根目录。
#
# 1) logs/*.log
#    删除「最后修改时间」早于 DAYS_LOGS 天的文件（默认 7 天）。
#
# 2) reports/*.txt
#    删除早于 DAYS_REPORTS_TXT 天的文本报告（默认 30 天）。
#
# 3) data/*.txt（若有零散 txt）
#    删除早于 DAYS_DATA_TXT 天的文件（默认 30 天）。
#
# 4) data/ 下按批次落盘的 JSON（仅匹配下列前缀/模式，且早于 DAYS_DATA_JSON 天）
#    - predictions_*.json, summary_*.json, reconcile_*.json, day_review_*.json
#    - validation_metrics_*.json
#    默认保留 90 天内的文件；超期删除。
#
# 【永不删除】下列迭代与状态文件（不在上述模式中）：
#    - reconcile_history.jsonl   （复盘滚动历史，持续迭代依赖）
#    - iteration_state.json
#    - iteration_briefing.txt
#
# 修改默认天数：可导出环境变量覆盖，例如：
#   DAYS_LOGS=14 DAYS_DATA_JSON=180 ./scripts/cleanup_stock_system.sh
# =============================================================================
set -euo pipefail

ROOT="${STOCK_SYSTEM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

DAYS_LOGS="${DAYS_LOGS:-7}"
DAYS_REPORTS_TXT="${DAYS_REPORTS_TXT:-30}"
DAYS_DATA_TXT="${DAYS_DATA_TXT:-30}"
DAYS_DATA_JSON="${DAYS_DATA_JSON:-90}"

mkdir -p logs reports data

n=0

# 1) logs
if [[ -d logs ]]; then
  while IFS= read -r -d '' f; do
    rm -f "$f"
    n=$((n + 1))
  done < <(find logs -maxdepth 1 -type f -name '*.log' -mtime "+${DAYS_LOGS}" -print0 2>/dev/null || true)
fi

# 2) reports/*.txt
if [[ -d reports ]]; then
  while IFS= read -r -d '' f; do
    rm -f "$f"
    n=$((n + 1))
  done < <(find reports -maxdepth 1 -type f -name '*.txt' -mtime "+${DAYS_REPORTS_TXT}" -print0 2>/dev/null || true)
fi

# 3) data/*.txt
if [[ -d data ]]; then
  while IFS= read -r -d '' f; do
    rm -f "$f"
    n=$((n + 1))
  done < <(find data -maxdepth 1 -type f -name '*.txt' ! -name 'iteration_briefing.txt' -mtime "+${DAYS_DATA_TXT}" -print0 2>/dev/null || true)
fi

# 4) data/*.json 批次文件（reconcile_history.jsonl 等为非匹配名，不会被删）
if [[ -d data ]]; then
  for pat in \
    'predictions_*.json' \
    'summary_*.json' \
    'reconcile_*.json' \
    'day_review_*.json' \
    'validation_metrics_*.json'; do
    while IFS= read -r -d '' f; do
      rm -f "$f"
      n=$((n + 1))
    done < <(find data -maxdepth 1 -type f -name "$pat" -mtime "+${DAYS_DATA_JSON}" -print0 2>/dev/null || true)
  done
fi

echo "cleanup_stock_system: removed ${n} file(s) under ${ROOT} (DAYS_LOGS=${DAYS_LOGS} DAYS_REPORTS_TXT=${DAYS_REPORTS_TXT} DAYS_DATA_TXT=${DAYS_DATA_TXT} DAYS_DATA_JSON=${DAYS_DATA_JSON})"
