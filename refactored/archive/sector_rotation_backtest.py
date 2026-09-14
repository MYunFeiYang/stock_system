#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""行业动量轮动回测 —— 用池内成分股合成行业等权指数（零新数据依赖）。

机制先验（非本样本挖掘）：
  1. 行业动量是文献中最稳的动量形态（Moskowitz-Grinblatt 1999 行业动量优于个股动量）
  2. ETF 交易免印花税，成本比个股低
  3. 降频（周/月）大幅降低换手成本

口径（零未来函数）：
  - 行业日收益 = 成分股等权平均
  - 每 hold_freq 日调仓：选过去 lookback 日行业累计收益 TOP3，等权持有
  - 成本：双边 0.10%（ETF 佣金水平）× 期初换手率
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_relative_strength import load_pool, get_history  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COST_PCT = 0.10   # 双边 %/次（ETF 水平）
TOP_K = 3


def _tstat(xs):
    n = len(xs)
    if n < 2:
        return {"n": n, "mean": 0.0, "t": 0.0}
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return {"n": n, "mean": m, "t": round(m / (sd / math.sqrt(n)), 2) if sd else 0.0}


def main():
    stocks = load_pool()
    hist = {s["symbol"]: get_history(s["symbol"]) for s in stocks}
    sec_of = {s["symbol"]: s.get("sector", "未分类") for s in stocks}
    valid = {c: dict(h) for c, h in hist.items() if len(h) > 130}
    dates = sorted({d for v in valid.values() for d in v})
    idx = {d: i for i, d in enumerate(dates)}
    arrs = {c: {idx[d]: p for d, p in v.items()} for c, v in valid.items()}

    # 行业日收益矩阵
    sectors = sorted({sec_of[c] for c in valid})
    sec_ret = {sec: {} for sec in sectors}   # sec -> {t: 日收益%}
    for t in range(1, len(dates)):
        by = {}
        for c, a in arrs.items():
            if t in a and (t - 1) in a and a[t - 1] > 0:
                by.setdefault(sec_of[c], []).append(a[t] / a[t - 1] - 1)
        for sec, rs in by.items():
            sec_ret[sec][t] = st.mean(rs)

    print(f"行业数: {len(sectors)}, 交易日: {len(dates)}")
    print(f"{'频率':>6} {'lookback':>8} | {'年化%':>7} {'t':>6} {'超额年化%':>9} {'最大回撤%':>9} | n")
    results = []
    for hold, tag in ((5, "周频"), (21, "月频")):
        for lookback in (20, 60):
            prev_hold = None
            rets, bench = [], []
            eq = 1.0
            peak = 1.0
            mdd = 0.0
            for t in range(lookback, len(dates) - 1):
                if (t - lookback) % hold != 0:
                    continue
                # 过去 lookback 日行业累计收益
                perf = {}
                for sec in sectors:
                    cum = 1.0
                    ok = True
                    for tt in range(t - lookback + 1, t + 1):
                        r = sec_ret[sec].get(tt)
                        if r is None:
                            ok = False
                            break
                        cum *= (1 + r)
                    if ok:
                        perf[sec] = cum - 1
                if len(perf) < 5:
                    continue
                top = sorted(perf, key=lambda s: -perf[s])[:TOP_K]
                # 持有至下一调仓日的收益 + 成本
                seg_rets = []
                for tt in range(t + 1, min(t + hold + 1, len(dates))):
                    rs = [sec_ret[s].get(tt) for s in top]
                    rs = [r for r in rs if r is not None]
                    if rs:
                        seg_rets.append(st.mean(rs))
                if not seg_rets:
                    continue
                period_ret = 1.0
                for r in seg_rets:
                    period_ret *= (1 + r)
                period_ret -= 1
                # 换手成本
                cost = 0.0
                if prev_hold is not None:
                    overlap = len(set(top) & set(prev_hold)) / TOP_K
                    cost = COST_PCT / 100 * (1 - overlap)
                prev_hold = top
                net = (1 + period_ret) * (1 - cost) - 1
                rets.append(net * 100)
                eq *= (1 + net)
                peak = max(peak, eq)
                mdd = min(mdd, eq / peak - 1)
                # 基准：全行业等权
                all_r = [sec_ret[s].get(tt) for s in sectors
                         for tt in range(t + 1, min(t + hold + 1, len(dates)))
                         if sec_ret[s].get(tt) is not None]
                if all_r:
                    bench.append(st.mean(all_r) * hold * 100)
            s = _tstat(rets)
            b = _tstat(bench)
            ann = s["mean"] * (252 / hold)
            bann = b["mean"] * (252 / hold)
            print(f"{tag:>6} {lookback:>8} | {ann:>+7.1f} {s['t']:>+6.2f} "
                  f"{ann - bann:>+9.1f} {mdd * 100:>+9.1f} | {s['n']}")
            results.append({"freq": tag, "lookback": lookback, "annual_pct": round(ann, 1),
                            "t": s["t"], "excess_annual_pct": round(ann - bann, 1),
                            "mdd_pct": round(mdd * 100, 1), "n": s["n"]})

    (ROOT / "data" / "sector_rotation_report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已写 data/sector_rotation_report.json")


if __name__ == "__main__":
    main()
