#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预测计分卡 —— 每日自动核对相对强弱预测的次日实际表现。

口径（零未来函数）：
  - 每次 TOP10 预测落盘于 data/rs_prediction_log.jsonl（按 pred_date 去重）
  - 核对：pred_date 次日，TOP10 平均收益 − 全池等权基准 = 超额
  - 滚动：全历史 + 最近 20 次预测的均值 / t 值 / 命中率

建议级别（数据驱动，写死防自欺，动态计算不硬编码）：
  - rolling_n >= 20 且 rolling_t > +2 且 mean > 0 → BUY_LEAN（买入倾向，TOP 列表可作候选）
  - rolling_n >= 20 且 rolling_t < -2            → AVOID（回避提示）
  - 其余                                         → OBSERVE（观察：预测尚未被实证）

由 predict_relative_strength.py 在每日预测刷新后自动调用。
"""
import datetime
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_relative_strength import load_pool, get_history  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "data" / "rs_prediction_log.jsonl"
OUT = ROOT / "data" / "rs_prediction_scorecard.json"
ROLL_N = 20          # 滚动窗口：最近 20 次预测
MIN_N_FOR_ADVICE = 20  # 建议级别所需最少核对次数
T_GATE = 2.0


def _tstat(xs):
    n = len(xs)
    if n < 2:
        return {"n": n, "mean": 0.0, "t": 0.0}
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    t = m / (sd / math.sqrt(n)) if sd else 0.0
    return {"n": n, "mean": round(m, 4), "t": round(t, 2)}


def _advice_level(roll: dict) -> str:
    if roll["n"] < MIN_N_FOR_ADVICE:
        return "OBSERVE"
    if roll["t"] > T_GATE and roll["mean"] > 0:
        return "BUY_LEAN"
    if roll["t"] < -T_GATE:
        return "AVOID"
    return "OBSERVE"


ADVICE_TEXT = {
    "BUY_LEAN": "买入倾向",
    "AVOID": "回避提示",
    "OBSERVE": "观察（预测尚未被实证）",
}


def update_scorecard() -> dict:
    stocks = load_pool()
    hist = {s["symbol"]: get_history(s["symbol"]) for s in stocks}
    valid = {c: dict(h) for c, h in hist.items() if len(h) > 5}
    dates = sorted({d for v in valid.values() for d in v})
    idx = {d: i for i, d in enumerate(dates)}
    arrs = {c: {idx[d]: p for d, p in v.items()} for c, v in valid.items()}

    if not LOG.exists():
        return {"n_checked": 0, "advice_level": "OBSERVE",
                "advice_text": ADVICE_TEXT["OBSERVE"]}

    records = []
    for ln in LOG.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(ln))
        except Exception:
            pass

    checked = []
    for r in records:
        pd = r.get("pred_date")
        top = r.get("top") or []
        if not pd or pd not in idx:
            continue
        t = idx[pd]
        if t + 1 >= len(dates):
            continue   # 次日未收盘，无法核对
        nxt = dates[t + 1]
        rets, bench = [], []
        for c, a in arrs.items():
            if t in a and (t + 1) in a and a[t] > 0:
                rets.append((c, a[t + 1] / a[t] - 1))
                bench.append(a[t + 1] / a[t] - 1)
        if len(rets) < 20 or not top:
            continue
        amap = dict(rets)
        top_rets = [amap[c] for c in top if c in amap]
        if len(top_rets) < 5:
            continue
        top_m = sum(top_rets) / len(top_rets)
        bench_m = sum(bench) / len(bench)
        hits = sum(1 for x in top_rets if x > bench_m)
        checked.append({
            "pred_date": pd, "checked_date": nxt,
            "top_ret_pct": round(top_m * 100, 3),
            "bench_ret_pct": round(bench_m * 100, 3),
            "excess_pct": round((top_m - bench_m) * 100, 3),
            "hit_rate": round(hits / len(top_rets), 2),
        })

    all_ex = [c["excess_pct"] for c in checked]
    roll_ex = [c["excess_pct"] for c in checked[-ROLL_N:]]
    all_stat = _tstat(all_ex)
    roll_stat = _tstat(roll_ex)
    level = _advice_level(roll_stat)

    out = {
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_logged": len(records),
        "n_checked": len(checked),
        "all_time": {**all_stat,
                     "hit_rate": round(sum(c["hit_rate"] for c in checked) / len(checked), 3)
                     if checked else None},
        "rolling": {**roll_stat,
                    "hit_rate": round(sum(c["hit_rate"] for c in checked[-ROLL_N:])
                                      / len(roll_ex), 3) if roll_ex else None},
        "advice_level": level,
        "advice_text": ADVICE_TEXT[level],
        "recent": checked[-5:],
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main():
    sc = update_scorecard()
    print(json.dumps({k: v for k, v in sc.items() if k != "recent"},
                     ensure_ascii=False, indent=1))
    for c in sc.get("recent", []):
        print(f"  {c['pred_date']} -> {c['checked_date']}: "
              f"TOP超额 {c['excess_pct']:+.2f}% 命中 {c['hit_rate']:.0%}")


if __name__ == "__main__":
    main()
