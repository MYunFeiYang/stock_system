#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""相对强弱预测的参数平面扫描 + 成本敏感性（防挑参数，看分布）。

网格: lookback(20/40/60/120) x decile(5/10) x 成本(0 / 0.10%按换手)
输出: data/rs_param_sweep.json —— 供早报动态引用「X/Y 显著」，
     禁止在推送文案中硬编码这些数字（防数字腐烂）。
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_relative_strength import load_pool, get_history, _tstat  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOOKBACKS = (20, 40, 60, 120)
NDECS = (5, 10)
COSTS = (0.0, 0.10)   # %/次, 按换手率折算


def main():
    stocks = load_pool()
    hist = {s["symbol"]: get_history(s["symbol"]) for s in stocks}
    sec_of = {s["symbol"]: s.get("sector", "未分类") for s in stocks}

    valid = {c: dict(h) for c, h in hist.items() if len(h) > 130}
    dates = sorted({d for v in valid.values() for d in v})
    idx = {d: i for i, d in enumerate(dates)}
    arrs = {c: {idx[d]: p for d, p in v.items()} for c, v in valid.items()}

    cells = []
    for lb in LOOKBACKS:
        for ndec in NDECS:
            for cost in COSTS:
                ls_rets, ex_rets = [], []
                turn = {10: 0.12, 5: 0.06}.get(ndec, 0.12)
                cday = cost / 100 * turn * 2
                for t in range(lb, len(dates) - 1):
                    moms = {c: a[t] / a[t - lb] - 1 for c, a in arrs.items()
                            if t in a and (t - lb) in a and (t + 1) in a
                            and a[t - lb] > 0}
                    if len(moms) < 20:
                        continue
                    by = {}
                    for c, m in moms.items():
                        by.setdefault(sec_of[c], []).append(m)
                    sm = {k: sum(v) / len(v) for k, v in by.items()}
                    sc = {c: m - sm[sec_of[c]] for c, m in moms.items()}
                    ranked = sorted(sc, key=lambda c: sc[c])
                    k = max(1, len(ranked) // ndec)
                    top, bot = ranked[-k:], ranked[:k]
                    rt = {c: arrs[c][t + 1] / arrs[c][t] - 1 for c in moms}
                    ls_rets.append((st.mean(rt[c] for c in top)
                                    - st.mean(rt[c] for c in bot) - cday) * 100)
                    ex_rets.append((st.mean(rt[c] for c in top)
                                    - st.mean(rt.values()) - cday) * 100)
                cells.append({
                    "lookback": lb, "ndec": ndec, "cost_pct": cost, "n": len(ls_rets),
                    "ls_t": _tstat(ls_rets)["t"],
                    "ls_annual_pct": round(_tstat(ls_rets)["mean"] * 252, 1),
                    "excess_t": _tstat(ex_rets)["t"],
                    "excess_annual_pct": round(_tstat(ex_rets)["mean"] * 252, 1),
                })

    cost_cells = [c for c in cells if c["cost_pct"] > 0]
    summary = {
        "n_cells": len(cost_cells),
        "ls_significant": sum(1 for c in cost_cells if abs(c["ls_t"]) > 2),
        "excess_significant": sum(1 for c in cost_cells if c["excess_t"] > 2),
        "excess_positive": sum(1 for c in cost_cells if c["excess_t"] > 0),
        "excess_t_range": [min(c["excess_t"] for c in cost_cells),
                           max(c["excess_t"] for c in cost_cells)],
    }
    out = {"cells": cells, "summary": summary}
    f = ROOT / "data" / "rs_param_sweep.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("已写:", f)


if __name__ == "__main__":
    main()
