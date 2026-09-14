#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""动态目标比例：便宜时多买、贵时少买（回应"点是否合适"的质疑）。

背景：固定 60/40 年度再平衡只保证"时间到"，不保证"点合适"。本脚本检验
**温和的估值倾斜**是否优于固定比例——注意这与已被证伪的"双均线全进全出
择时"不同：这里是温和调整目标比例（如 60%→50%/70%），不是 0/100 切换。

贵贱代理（免费可得）：股票指数相对其自身 5 年(1220日)均线的偏离 z-score。
动态目标 = 基准 - k × z，限幅 [40%, 80%]（风险梯度可控）。
k 用参数平面（0.10/0.15/0.20），不挑参数。
"""
import json
import os
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "style_index_cache_6000"
OUT = ROOT / "data" / "timing_tilt_report.json"
COST = 0.0005
DIV = float(os.environ.get("STOCK_DIV", "2")) / 100
DIV_D = (1 + DIV) ** (1 / 244) - 1
BASE = 0.60
LO, HI = 0.40, 0.80
MA_WIN = 1220          # 5 年
REBAL = 244            # 年度


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
    prices = [stock[d] for d in dates]
    print(f"样本 {dates[0]}~{dates[-1]} ({len(dates)/244:.1f}年), "
          f"股息+{DIV*100:.0f}%, 均线{MA_WIN}日", flush=True)

    def zscore(i):
        """价格相对 5 年均线的偏离（标准化）。正=贵，负=便宜。"""
        if i < MA_WIN:
            return None
        win = prices[i - MA_WIN:i + 1]
        mu = st.mean(win)
        sd = st.pstdev(win)
        if sd <= 0:
            return None
        return (prices[i] - mu) / sd

    def sim(k):
        ws, wb = BASE, 1 - BASE
        eq, eqs, rets, n, targets = 1.0, [], [], 0, []
        for i in range(1, len(dates)):
            ws *= (1 + sret.get(i, 0.0))
            wb *= (1 + bret.get(i, 0.0))
            tot = ws + wb
            ws, wb = ws / tot, wb / tot
            dret = tot - 1.0
            if i % REBAL == 0:
                z = zscore(i)
                tgt = BASE if (k == 0 or z is None) else \
                    min(HI, max(LO, BASE - k * z))
                targets.append(round(tgt, 3))
                turn = abs(ws - tgt)
                eq *= (1 - turn * COST * 2)
                ws, wb = tgt, 1 - tgt
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
        return {"k": k, "cagr": round(cagr, 2), "mdd": round(mdd * 100, 1),
                "sharpe": round(cagr / vol, 2) if vol else 0,
                "n_rebal": n,
                "tgt_range": [min(targets), max(targets)] if targets else []}

    rows = []
    print(f"\n{'k(倾斜强度)':<12}{'年化':>8}{'回撤':>9}{'夏普':>7}"
          f"{'目标比例区间':>16}", flush=True)
    for k in (0.0, 0.10, 0.15, 0.20):
        r = sim(k)
        rows.append(r)
        rng = f"{r['tgt_range'][0]:.0%}~{r['tgt_range'][1]:.0%}" \
            if r["tgt_range"] else "固定60%"
        print(f"{k:<12}{r['cagr']:>+7.2f}%{r['mdd']:>8.1f}%"
              f"{r['sharpe']:>7.2f}{rng:>16}", flush=True)
    best = max(rows, key=lambda r: r["cagr"])
    base = rows[0]
    print(f"\n基准(固定60%): {base['cagr']:+.2f}% 回撤{base['mdd']}% "
          f"夏普{base['sharpe']}", flush=True)
    print(f"最佳( k={best['k']}): {best['cagr']:+.2f}% 回撤{best['mdd']}% "
          f"夏普{best['sharpe']}", flush=True)
    if best["cagr"] > base["cagr"] and best["mdd"] >= base["mdd"]:
        print("→ 动态倾斜**有效**：收益更高且回撤不恶化", flush=True)
    elif best["cagr"] > base["cagr"]:
        print("→ 收益更高但回撤变差，需权衡", flush=True)
    else:
        print("→ 动态倾斜**无效**：固定 60% 更优（与价格择时已被证伪一致）",
              flush=True)
    OUT.write_text(json.dumps({"start": dates[0], "end": dates[-1],
                               "rows": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"已写 {OUT.name}", flush=True)


if __name__ == "__main__":
    main()
