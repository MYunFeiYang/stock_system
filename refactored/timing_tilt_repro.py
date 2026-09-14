#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第三方对齐复算：严格逐行复刻 timing_tilt_validate.py 口径，独立重跑，
验证项目结论(t=3.04 / CAGR10.64% / MDD-28.4%)是否可复现。
若复现→项目自体可信，子代理差异=口径不同；若不复现→项目脚本有 bug。
"""
import json
import os
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "style_index_cache_6000"
COST = 0.0005
DIV = float(os.environ.get("STOCK_DIV", "2")) / 100
DIV_D = (1 + DIV) ** (1 / 244) - 1
BASE = 0.60
LO, HI = 0.40, 0.80
MA_WIN = 1220
REBAL = 244


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

    def zscore(i):
        if i < MA_WIN:
            return None
        win = prices[i - MA_WIN:i + 1]
        mu, sd = st.mean(win), st.pstdev(win)
        return (prices[i] - mu) / sd if sd > 0 else None

    def curve(k):
        ws, wb = BASE, 1 - BASE
        eqs = []
        for i in range(1, len(dates)):
            ws *= (1 + sret.get(i, 0.0))
            wb *= (1 + bret.get(i, 0.0))
            tot = ws + wb
            ws, wb = ws / tot, wb / tot
            if i % REBAL == 0:
                z = zscore(i)
                tgt = BASE if (k == 0 or z is None) else min(HI, max(LO, BASE - k * z))
                turn = abs(ws - tgt)
                eq = 1 - turn * COST * 2
                ws, wb = tgt, 1 - tgt
            else:
                eq = 1.0
            eq *= (1 + tot - 1.0)
            eqs.append(eq)
        return eqs

    def stats(eqs):
        eq = 1.0
        for e in eqs:
            eq *= e
        years = len(dates) / 244
        cagr = (eq ** (1 / years) - 1) * 100
        peak, mdd, e = 1.0, 0.0, 1.0
        for r in eqs:
            e *= r
            peak = max(peak, e)
            mdd = min(mdd, e / peak - 1)
        vol = st.pstdev([(x - 1) * 100 for x in eqs]) * (252 ** 0.5)
        return cagr, mdd * 100, vol

    K = 0.15
    tilt = curve(K)
    fixed = curve(0.0)
    tc, tm, tv = stats(tilt)
    fc, fm, fv = stats(fixed)
    excess = [t / f - 1 for t, f in zip(tilt, fixed)]
    ex = [x * 100 for x in excess]
    mu = st.mean(ex)
    t = mu / (st.pstdev(ex) / (len(ex) ** 0.5))
    print(f"对齐复算 样本 {dates[0]}~{dates[-1]} ({len(dates)/244:.1f}年)")
    print(f"倾斜(k=0.15): CAGR {tc:+.2f}%  MDD {tm:.1f}%  夏普 {tc/tv:.2f}")
    print(f"固定60/40  : CAGR {fc:+.2f}%  MDD {fm:.1f}%  夏普 {fc/fv:.2f}")
    print(f"逐日超额 t = {t:+.2f}  均值 {mu:+.4f}%/日")
    print("对照项目(timing_tilt_validate.json): 期望 t≈+3.04 CAGR≈+10.64% MDD≈-28.4%")
    print("→ 复现成功" if abs(t - 3.04) < 0.5 and abs(tc - 10.64) < 1.0 and abs(tm - (-28.4)) < 3 else "→ 不复现，项目脚本或口径存疑")


if __name__ == "__main__":
    main()
