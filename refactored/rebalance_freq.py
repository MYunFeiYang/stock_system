#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""调仓频率 vs 净收益：买卖点到底该多久一次？

用户诉求"预测正确的买入卖出点"。预测路线已全部证伪，但买卖点也可以来自
**规则**（再平衡）。本脚本回答：规则型买卖点多频繁最优？
频率从日频到三年一次全覆盖，扣 ETF 成本后对比净收益。
"""
import json
import os
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "style_index_cache_6000"
OUT = ROOT / "data" / "rebalance_freq_report.json"
COST = 0.0005          # ETF 单边 ≈0.05%（免印花税）
DIV = float(os.environ.get("STOCK_DIV", "2")) / 100
DIV_D = (1 + DIV) ** (1 / 244) - 1


def load(sym):
    cf = CACHE / f"{sym}_6000.json"
    return dict((d, c) for d, c in json.loads(cf.read_text(encoding="utf-8")))


def main():
    stock, bond = load("sh000300"), load("sh000012")
    start = max(min(stock), min(bond))
    dates = sorted(d for d in stock if d >= start)
    sret, bret = {}, {}
    for i in range(1, len(dates)):
        d0, d1 = dates[i - 1], dates[i]
        if d0 in stock and d1 in stock and stock[d0] > 0:
            sret[i] = stock[d1] / stock[d0] - 1 + DIV_D
        if d0 in bond and d1 in bond and bond[d0] > 0:
            bret[i] = bond[d1] / bond[d0] - 1
    print(f"样本 {dates[0]}~{dates[-1]} ({len(dates)/244:.1f}年), "
          f"股票端补股息{DIV*100:.0f}%", flush=True)

    def sim(w_stock, every):
        ws, wb = w_stock, 1 - w_stock
        eq, eqs, rets, n = 1.0, [], [], 0
        for i in range(1, len(dates)):
            ws *= (1 + sret.get(i, 0.0))
            wb *= (1 + bret.get(i, 0.0))
            tot = ws + wb
            ws, wb = ws / tot, wb / tot
            dret = tot - 1.0
            if i % every == 0:
                turn = abs(ws - w_stock)
                eq *= (1 - turn * COST * 2)
                ws, wb = w_stock, 1 - w_stock
                n += 1
            eq *= (1 + dret)
            eqs.append(eq)
            rets.append(dret * 100)
        peak, mdd = 1.0, 0.0
        for e in eqs:
            peak = max(peak, e)
            mdd = min(mdd, e / peak - 1)
        years = len(dates) / 244
        cagr = (eq ** (1 / years) - 1) * 100
        vol = st.pstdev(rets) * (252 ** 0.5)
        return {"cagr": round(cagr, 2), "mdd": round(mdd * 100, 1),
                "sharpe": round(cagr / vol, 2) if vol else 0, "n_rebal": n}

    cases = [("日频(每天)", 1), ("周频(5日)", 5), ("月频(21日)", 21),
             ("季频(61日)", 61), ("半年(122日)", 122), ("年度(244日)", 244),
             ("两年(488日)", 488), ("三年(732日)", 732)]
    rows = []
    print(f"\n{'调仓频率':<14}{'年化':>8}{'回撤':>9}{'夏普':>7}{'调仓次数':>9}",
          flush=True)
    for label, ev in cases:
        r = sim(0.6, ev)
        rows.append({"freq": label, "every": ev, **r})
        print(f"{label:<14}{r['cagr']:>+7.2f}%{r['mdd']:>8.1f}%"
              f"{r['sharpe']:>7.2f}{r['n_rebal']:>9}", flush=True)
    best = max(rows, key=lambda r: r["cagr"])
    print(f"\n年化最高: {best['freq']} ({best['cagr']:+.2f}%, "
          f"{best['n_rebal']}次调仓)", flush=True)
    bs = max(rows, key=lambda r: r["sharpe"])
    print(f"夏普最高: {bs['freq']} (夏普{bs['sharpe']})", flush=True)
    OUT.write_text(json.dumps({"start": dates[0], "end": dates[-1],
                               "rows": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"已写 {OUT.name}", flush=True)


if __name__ == "__main__":
    main()
