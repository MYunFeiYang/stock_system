#!/usr/bin/env bash
# 直跑股票分析：Python 流水线内通过 `openclaw agent` 取数（非频道 agentTurn）。crontab 需 PATH 含 openclaw。
set -euo pipefail
ROOT="${STOCK_SYSTEM_ROOT:-${OPENCLAW_HOME:-$HOME/.openclaw}/workspace-stock/stock_system}"
cd "$ROOT"
exec python3 refactored/openclaw_cron_analyzer.py "${1:-morning}"
