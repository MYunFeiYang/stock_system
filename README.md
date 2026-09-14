# A 股分析（OpenClaw 主链路）

**唯一推荐入口**：`refactored/openclaw_cron_analyzer.py`

参数：`morning` | `afternoon` | `evening` | `weekly` | `reconcile` | `day_review` | **`post_close`**（定时推荐：先 `reconcile` 再 `day_review`，汇总与自校准立刻用到刚写入的复盘）。

**OpenClaw Cron（工作日）**：**08:00 `morning`** → **18:10 `post_close`**（一次完成收盘复盘 + 全日汇总 + 规则自校准；晚于 18:00 拉价更稳）；另每日 **02:00** 运行 `scripts/cleanup_stock_system.sh`（与预测无关）。

**定期清理规则**（可调环境变量覆盖天数）：

| 路径 | 规则 | 默认 |
|------|------|------|
| `logs/*.log` | 超过 N 天未修改则删除 | `DAYS_LOGS=7` |
| `reports/*.txt` | 超过 N 天未修改则删除 | `DAYS_REPORTS_TXT=30` |
| `data/*.txt`（不含 `iteration_briefing.txt`） | 超过 N 天未修改则删除 | `DAYS_DATA_TXT=30` |
| `data/predictions_*.json`、`summary_*.json`、`reconcile_*.json`、`day_review_*.json`、`validation_metrics_*.json` | 超过 N 天未修改则删除 | `DAYS_DATA_JSON=90` |
| `reconcile_history.jsonl`、`iteration_state.json` | **不删除**（无匹配模式） | — |
| `data/iteration_briefing.txt` | **不删除**（已从 `data/*.txt` 清理中排除） | — |

`afternoon` / `evening` / `weekly` 可手动执行，已不从 Cron 触发。

```bash
cd /path/to/stock_system
python3 refactored/openclaw_cron_analyzer.py morning
```

- 预测与总结：`refactored/predict_then_summarize.py`
- 取数：`refactored/openclaw_search_provider.py`（本机 `openclaw agent`）
- 股票池：`config/stock_pool.json`
- 系统 crontab 直跑（不经 OpenClaw `agentTurn`）：`scripts/run_stock_cron.sh`

环境变量：`STOCK_SYSTEM_ROOT`、`OPENCLAW_BIN`、`OPENCLAW_AGENT_TIMEOUT`、`STOCK_OPENCLAW_CACHE_SEC`（见仓库根目录 `SETUP.md`）。

## LLM 预测（OpenRouter）

设置以下环境变量可将预测引擎从公式打分切换为 LLM 预测：

```bash
export STOCK_USE_LLM=1                          # 启用 LLM 预测
export OPENROUTER_API_KEY="sk-or-v1-xxxx"        # OpenRouter API Key
export OPENROUTER_MODEL="deepseek/deepseek-chat"  # 模型（可选，默认 deepseek）
```

不设置 `STOCK_USE_LLM` 或未提供 API Key 时，系统**自动回退**到公式打分，行为与之前完全一致。

**LLM 预测流程**：
1. 批量模式（推荐）：一次 API 调用预测组合内所有股票，LLM 可做相对比较
2. 单只模式：批量失败时降级到逐只 LLM 预测
3. 公式 fallback：LLM 全部失败时自动回退到分段线性公式打分

**推荐模型**（按性价比排序）：

| 模型 | 成本 (per 1M tokens) | 特点 |
|------|---------------------|------|
| `deepseek/deepseek-chat` | $0.14 / $0.28 | 中文最佳，最便宜 |
| `qwen/qwen-3-32b` | 免费 | 开源中文强，零成本 |
| `google/gemini-2.0-flash-001` | 免费额度大 | 速度最快 |
| `anthropic/claude-3.5-haiku` | $1.0 / $5.0 | 分析能力最强 |

- 取数仍走 akshare（新浪 HTTP），LLM **只做预测**，不做数据抓取
- LLM 模块 `refactored/llm_predictor.py` 纯标准库，零第三方依赖

**免费强化 OpenClaw 搜索（仍为 DuckDuckGo）**：在 `openclaw.json` 为 `duckduckgo` 配 `region: cn-zh`，并视需要调 `tools.web.search` 的 `maxResults` / `cacheTtlMinutes`。片段见根目录 `SETUP.md`（含敏感项的配置请勿提交到公开仓库）。

落盘目录：`data/`、`reports/`、`logs/`。

## 规则自校准（不调模型）

每个交易日 **`day_review` 结束后** 会根据 `data/reconcile_history.jsonl` 滚动统计，更新 **`config/calibration_overrides.json`**（信号阈值 `signal_thresholds`、综合分权重 `score_weights`）。**次日早盘** `StockAnalyzer.analyze` 开头会重新加载该文件。该文件已加入 `.gitignore`，本地自动生成；字段格式可参考同目录 **`calibration_overrides.example.json`**。

- 样本过少（有效方向样本不足 24）时**只写 meta、不改数值**。
- 滚动统计对交易日做 **指数衰减加权**（越近权重越大），并按 **强烈买入/普通买入、强烈卖出/普通卖出** 分拆调门槛，与复盘强档规则一致。
- 自校准超参集中在 **`auto_calibration`**（与代码默认值合并，写入 overrides 时会落盘；见 `calibration_overrides.example.json`）：衰减系数、回溯交易日数、样本门槛、单步 `step_th`/`step_w`、误判率高低阈值、买卖误判对比倍数、历史 jsonl 最大行数等。
- 单次调整幅度有上限，且阈值保持 `strong_buy > buy > hold > sell`。
- 预测落盘 JSON 中带 `calibration` 字段，便于核对当时使用的档位。

手动试跑：`python3 refactored/auto_calibration.py`（可选参数：stock_system 根目录）。

### 准确率相关（已落地）

- **reconcile**：中性带按早盘 `|change_percent|` **自适应**；**强烈买入/强烈卖出** 要求当日区间涨跌超过「强档带」才算方向一致。可选 **`RECONCILE_BENCHMARK_RETURN_PCT`**（例如当日沪深300涨跌幅），用 **个股涨跌减大盘** 做方向判定。
- **打分**：技术面/基本面等子项改为 **分段线性**，减轻阶梯边界抖动。
- **信号**：`calibration_overrides.json` 里 **`accuracy_tuning.signal_margin`**（示例见 `calibration_overrides.example.json`）在阈值贴边时把买/卖/强档向 **持有或弱一档** 收敛。
- **收盘预测**：`evening_optimizer` 与早盘共用同一套 **阈值、权重、signal_margin**（仍保留时间衰减）。
- **自校准**：除买/卖外，**持有误判** 也会驱动收窄/放宽持有带（`hold`/`buy` 边界）。
