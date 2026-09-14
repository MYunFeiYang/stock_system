#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""估值倾斜修正验证：修正 timing_tilt_validate.py 的两处统计缺陷。
缺陷1: 超额用 净值比率差(tilt/fixed-1) 且普通t → 在自正相关下膨胀显著性。
       修正: 超额用 日收益差(arithmetic)，t 用 Newey-West 调整。
缺陷2: 权重实现对照审计口径(漂移+年度再平衡)，与 audit.py 每日恒定权重区分开。
目的: 给出估值倾斜真实的 per-day alpha 显著性，供拍板。
"""
import json
import os
import math
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
    return dict((d, c) for d, c in json.loads(
        (CACHE / f"{sym}_6000.json").read_text(encoding="utf-8")))


def tstat_nw(exc):
    N = len(exc)
    mu = st.mean(exc)
    sd = (sum((x - mu) ** 2 for x in exc) / (N - 1)) ** 0.5
    t_ord = mu / (sd / math.sqrt(N))
    q = max(1, int(4 * (N / 100) ** (2 / 9)))
    g0 = sum((x - mu) ** 2 for x in exc) / N
    sac = g0
    for kk in range(1, q + 1):
        gk = sum((exc[t] - mu) * (exc[t - kk] - mu) for t in range(kk, N)) / N
        sac += 2 * gk * (1 - kk / (q + 1))
    t_nw = mu / math.sqrt(sac / N)
    return t_ord, t_nw, q


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

    # 正确口径: 漂移+年度再平衡, 日收益序列
    def daily_rets(k):
        ws, wb = BASE, 1 - BASE
        rs = [0.0] * len(dates)
        for i in range(1, len(dates)):
            open_w = ws
            r = open_w * sret.get(i, 0.0) + (1 - open_w) * bret.get(i, 0.0)
            ws *= (1 + sret.get(i, 0.0))
            wb *= (1 + bret.get(i, 0.0))
            tot = ws + wb
            ws, wb = ws / tot, wb / tot
            if i % REBAL == 0:
                z = zscore(i)
                tgt = BASE if (k == 0 or z is None) else min(HI, max(LO, BASE - k * z))
                cost = abs(ws - tgt) * COST * 2
                r -= cost
                ws, wb = tgt, 1 - tgt
            rs[i] = r
        return rs

    def mdd(rs):
        g, peak, md = 1.0, 1.0, 0.0
        for x in rs[1:]:
            g *= (1 + x)
            peak = max(peak, g)
            md = min(md, g / peak - 1)
        return md

    def cagr(rs):
        g = 1.0
        for x in rs[1:]:
            g *= (1 + x)
        yrs = (len(rs) - 1) / 244
        return g ** (1 / yrs) - 1

    tilt = daily_rets(0.15)
    fixed = daily_rets(0.0)
    print("=== 正确口径(日收益差 + Newey-West) ===")
    print(f"TILT  CAGR {cagr(tilt)*100:+.2f}%  MDD {mdd(tilt)*100:.1f}%")
    print(f"FIXED CAGR {cagr(fixed)*100:+.2f}%  MDD {mdd(fixed)*100:.1f}%")
    exc = [tilt[i] - fixed[i] for i in range(1, len(dates))]
    t_o, t_nw, q = tstat_nw(exc)
    cut = next(i for i, d in enumerate(dates) if d >= "2015-01-01")
    pre = exc[:cut - 1]
    post = exc[cut - 1:]
    po, pnw, _ = tstat_nw(pre)
    oo, onw, _ = tstat_nw(post)
    print(f"EXCESS(收益差) mean {st.mean(exc)*100:.4f}%/d  t_ord={t_o:.2f}  "
          f"t_NW(q={q})={t_nw:.2f}")
    print(f"  2015前 t_ord={po:.2f} t_NW={pnw:.2f} | 2015后 t_ord={oo:.2f} t_NW={onw:.2f}")

    # 旧口径对照(净值比率差 + 普通t)
    def curve_eqs(k):
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
            eq *= tot
            eqs.append(eq)
        return eqs

    te, fe = curve_eqs(0.15), curve_eqs(0.0)
    ex2 = [t / f - 1 for t, f in zip(te, fe)]
    mu2 = st.mean(ex2)
    t2 = mu2 / (st.pstdev(ex2) / math.sqrt(len(ex2)))
    print(f"\n=== 旧口径(净值比率差 + 普通t, 即之前报的) ===")
    print(f"EXCESS mean {mu2*100:.4f}%/d  t_ord={t2:.2f}")
    print("\n→ 判定: 若 t_NW < 2, 旧口径 t 因自相关未调整而膨胀, 真实不显著")
    print(f"  结论: 真实 t_NW={t_nw:.2f} {'≥2 显著' if t_nw >= 2 else '<2 不显著(推翻上生产依据)'}")

    out = {
        "correct_excess_mean_pct": round(st.mean(exc) * 100, 4),
        "correct_t_ord": round(t_o, 2), "correct_t_nw": round(t_nw, 2),
        "correct_t_nw_pre": round(pnw, 2), "correct_t_nw_post": round(onw, 2),
        "old_t_ord": round(t2, 2),
        "verdict": "SIGNIFICANT" if t_nw >= 2 else "NOT_SIGNIFICANT_inflated_old_t",
    }
    (ROOT / "data" / "timing_tilt_correct.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已写 timing_tilt_correct.json")


if __name__ == "__main__":
    main()
