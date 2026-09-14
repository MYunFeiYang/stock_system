#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""历史预测回填 —— walk-forward 重放生成过去 N 日的 TOP10 预测并合并落盘。

目的：计分卡冷启动立刻有核对样本，不用干等 20 个交易日积累。
合法性：与实盘预测完全同构——每个 pred_date 只用该日及之前数据（零未来函数），
与 backtest 的 walk-forward 重放是同一性质，非"事后诸葛亮"。
回填记录带 type=backfill 标记，与实盘落盘区分但同样参与核对。
"""
import datetime
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_relative_strength import load_pool, get_history, MOM_LOOKBACK  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "data" / "rs_prediction_log.jsonl"
BACKFILL_DAYS = 60   # 回填最近 60 个交易日


def main():
    stocks = load_pool()
    hist = {s["symbol"]: get_history(s["symbol"]) for s in stocks}
    sec_of = {s["symbol"]: s.get("sector", "未分类") for s in stocks}
    valid = {c: dict(h) for c, h in hist.items() if len(h) > MOM_LOOKBACK + 5}
    dates = sorted({d for v in valid.values() for d in v})
    idx = {d: i for i, d in enumerate(dates)}
    arrs = {c: {idx[d]: p for d, p in v.items()} for c, v in valid.items()}
    name_of = {s["symbol"]: s["name"] for s in stocks}

    # 既有记录按 pred_date 索引
    recs = {}
    if LOG.exists():
        for ln in LOG.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(ln)
                recs[r["pred_date"]] = r
            except Exception:
                pass

    added = 0
    start = max(MOM_LOOKBACK + 1, len(dates) - 1 - BACKFILL_DAYS)
    for t in range(start, len(dates) - 1):
        pd = dates[t]
        if pd in recs and recs[pd].get("type") == "live":
            continue   # 不覆盖实盘记录
        moms = {}
        for c, a in arrs.items():
            if t in a and (t - MOM_LOOKBACK) in a and a[t - MOM_LOOKBACK] > 0:
                moms[c] = a[t] / a[t - MOM_LOOKBACK] - 1.0
        if len(moms) < 20:
            continue
        by = {}
        for c, m in moms.items():
            by.setdefault(sec_of[c], []).append(m)
        sm = {k: sum(v) / len(v) for k, v in by.items()}
        sc = {c: m - sm[sec_of[c]] for c, m in moms.items()}
        ranked = sorted(sc, key=lambda c: -sc[c])[:10]
        recs[pd] = {
            "pred_date": pd,
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "type": "backfill",
            "top": ranked,
        }
        added += 1

    out = sorted(recs.values(), key=lambda r: r["pred_date"])
    LOG.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out)
                   + "\n", encoding="utf-8")
    print(f"回填 {added} 条，日志共 {len(out)} 条 -> {LOG.name}")

    # 立即更新计分卡
    from rs_scorecard import update_scorecard
    sc = update_scorecard()
    print(f"计分卡: n_checked={sc['n_checked']} "
          f"rolling(t={sc['rolling'].get('t')}, mean={sc['rolling'].get('mean')}%/日, "
          f"hit={sc['rolling'].get('hit_rate')}) "
          f"建议级别={sc['advice_level']}")


if __name__ == "__main__":
    main()
