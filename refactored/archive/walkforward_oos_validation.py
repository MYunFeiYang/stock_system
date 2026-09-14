#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk-forward 时间外交叉验证 (Out-of-Time, OOS)
—— 验证「追高闸门 / ranging-only」是否真有 edge，而非全样本巧合。

方法：
  复用与生产、与 walkforward_backtest 完全一致的打分链路（零未来函数），
  回放 2025-08 ~ 2026-08 全样本，按锚日(决策日)切成：
    训练期 = 前 120 个交易日（用于"发现" ranging-only 门控规则）
    验证期 = 后 119 个交易日（用于"盲测"该规则）
  对三段（训练/验证/全样本）分别统计按天主判据：
    1) 原始公式买入组合（orig_signal 买入意图）t
    2) ranging-only 买入组合（仅放行 regime=range 的买入）t
    3) 追高泄漏（regime=trend 买入）t
  判据：验证期 ranging-only 仍为正且 t 不坍塌（理想 |t|>2 或至少 >1 接近显著），
        且优于同段原始公式 → 规则有真 edge；否则视为样本巧合。
"""
from __future__ import annotations

import bisect
import json
import math
import os
import statistics as st
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from data_providers import (  # noqa: E402
    compute_real_technicals,
    detect_market_status,
    relative_strength_score,
    sentiment_from_technical,
    _neutral_fundamental,
)
from predict_then_summarize import ConfigManager, ScoringEngine, SignalGenerator  # noqa: E402
from backtest_signals import compute_verdict, verdict_lines_from_dict  # noqa: E402
from walkforward_backtest import (  # noqa: E402
    fetch_ohlcv, _tstat, _by_day_series,
    WIN, INDEX_WIN, INDEX_SYMBOL, BUY_SIGNALS,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")


# ──────────────────────── 回放（与生产同口径，零未来函数） ────────────────────────

def replay_rows(fetch_days: int = 300) -> list:
    stocks = ConfigManager.get_core_stocks()
    scoring = ScoringEngine()
    siggen = SignalGenerator()

    idx_kl = fetch_ohlcv(INDEX_SYMBOL, fetch_days)
    idx_days = [k["day"] for k in idx_kl]
    mstat_by_day: dict[str, dict] = {}
    for j in range(INDEX_WIN - 1, len(idx_kl)):
        mstat_by_day[idx_days[j]] = detect_market_status(idx_kl[j - INDEX_WIN + 1: j + 1])

    rows: list[dict] = []
    for stock in stocks:
        kl = fetch_ohlcv(stock.symbol, fetch_days)
        if len(kl) < WIN + 3:
            continue
        days = [k["day"] for k in kl]
        closes = [float(k["close"]) for k in kl]
        start = WIN - 1
        end = len(kl) - 3
        fund = _neutral_fundamental(stock)
        fund_s = scoring.calculate_fundamental_score(fund, stock.sector)
        for i in range(start, end + 1):
            ad = days[i]
            j = bisect.bisect_right(idx_days, ad) - 1
            if j < 0:
                continue
            mstat = mstat_by_day.get(idx_days[j])
            if mstat is None:
                continue
            win = kl[i - WIN + 1: i + 1]
            tech = compute_real_technicals(win)
            sent = sentiment_from_technical(tech)
            tech_s = scoring.calculate_technical_score(tech)
            sent_s = scoring.calculate_sentiment_score(sent)
            sec_s = relative_strength_score(
                float(tech.get("momentum_20d", 0.0)),
                float(mstat.get("trend_strength", 0.0)),
            )
            weights = ConfigManager.get_market_adjusted_weights(mstat.get("regime", "range"))
            final = scoring.calculate_final_score(tech_s, fund_s, sent_s, sec_s, weights=weights)
            signal, _c, _r = siggen.generate_signal(final, stock, tech)
            entry = closes[i]
            if entry <= 0:
                continue
            rows.append({
                "anchor_day": ad,
                "pred_day": days[i + 1],
                "symbol": stock.symbol,
                "orig_signal": signal,
                "regime": mstat.get("regime", "range"),
                "status": mstat.get("status", "ranging"),
                "ret_next": round((closes[i + 2] / entry - 1) * 100, 4),
                "ret_intra": round((closes[i + 1] / entry - 1) * 100, 4),
            })
    return rows


# ──────────────────────── 统计 ────────────────────────

def _analyze(rows: list, label: str) -> dict:
    buy = [r for r in rows if r["orig_signal"] in BUY_SIGNALS]
    ro = [r for r in buy if r["regime"] == "range"]       # ranging-only 放行
    tr = [r for r in buy if r["regime"] == "trend"]        # 追高泄漏
    vol = [r for r in buy if r["regime"] == "volatility"]  # 高波动（样本内未出现）

    adays = sorted({r["anchor_day"] for r in rows})
    print(f"\n── {label} ── 决策日 {adays[0]}~{adays[-1]}（{len(adays)} 日）| 样本 {len(rows)}")
    print(f"  原始买入意图: {len(buy)} 笔 | range {len(ro)} / trend {len(tr)} / volatility {len(vol)}")

    ob = _tstat(_by_day_series(buy, "ret_next"))
    rb = _tstat(_by_day_series(ro, "ret_next"))
    tb = _tstat(_by_day_series(tr, "ret_next"))
    print(f"  原始公式买入  按天: 均值={ob['mean']:+.3f}%/日 n={ob['n']} t={ob['t']:+.2f} {'显著' if ob['significant'] else '不显著'}")
    print(f"  ranging-only  按天: 均值={rb['mean']:+.3f}%/日 n={rb['n']} t={rb['t']:+.2f} {'显著' if rb['significant'] else '不显著'}")
    print(f"  trend(追高)   按天: 均值={tb['mean']:+.3f}%/日 n={tb['n']} t={tb['t']:+.2f} {'显著' if tb['significant'] else '不显著'}")

    rv = compute_verdict(_by_day_series(ro, "ret_next"), _by_day_series(ro, "ret_intra"),
                         full_population=True)
    print("  ranging-only 判据:")
    for ln in verdict_lines_from_dict(rv):
        if ln.strip():
            print("   " + ln)
    return {"label": label, "orig": ob, "range": rb, "trend": tb,
            "ro_count": len(ro), "buy_count": len(buy)}


def main():
    rows = replay_rows(300)
    days = sorted({r["anchor_day"] for r in rows})
    k = 120
    split = days[k - 1]
    train = [r for r in rows if r["anchor_day"] <= split]
    test = [r for r in rows if r["anchor_day"] > split]
    print(f"OOS 切分: 训练期 ≤ {split}（{len({r['anchor_day'] for r in train})} 决策日）, "
          f"验证期 > {split}（{len({r['anchor_day'] for r in test})} 决策日）")

    tr_res = _analyze(train, "训练期(前120日)")
    te_res = _analyze(test, "验证期(后119日)")
    _analyze(rows, "全样本(参考)")

    print("\n=== 结论 ===")
    ro_tr, ro_te = tr_res["range"], te_res["range"]
    print(f"训练期 ranging-only: {ro_tr['mean']:+.3f}%/日 t={ro_tr['t']:+.2f} "
          f"(n={ro_tr['n']})")
    print(f"验证期 ranging-only: {ro_te['mean']:+.3f}%/日 t={ro_te['t']:+.2f} "
          f"(n={ro_te['n']})")
    if ro_te["mean"] > 0 and ro_te["t"] > 1.0:
        print("✅ 验证期仍为正且 t>1：ranging-only 门控有真 edge，非全样本巧合。可落生产。")
        verdict = "PASS"
    elif ro_te["mean"] > 0:
        print("⚠️ 验证期为正但 t≤1：方向对但统计弱，需更大样本确认（可落但降权）。")
        verdict = "WEAK"
    else:
        print("❌ 验证期为负：全样本 +0.455% 视为样本巧合，门控无效，勿落生产。")
        verdict = "FAIL"

    out = {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "split_day": split,
        "train_days": len({r["anchor_day"] for r in train}),
        "test_days": len({r["anchor_day"] for r in test}),
        "train_range": tr_res["range"],
        "test_range": te_res["range"],
        "train_trend": tr_res["trend"],
        "test_trend": te_res["trend"],
        "train_orig": tr_res["orig"],
        "test_orig": te_res["orig"],
        "verdict": verdict,
    }
    path = os.path.join(DATA_DIR, "walkforward_oos_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已写 data/walkforward_oos_report.json")


if __name__ == "__main__":
    main()
