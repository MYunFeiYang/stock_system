#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""申万行业指数 · 低频经典因子扫描（月频/季频）。

动机：此前所有验证都在日频/周频（过拟合工厂）。而 A 股文献中长期有效的
真因子——短期反转、低波动——都是低频（月/季调仓）形态，且有经济学先验
（反转=流动性冲击/过度反应；低波=彩票偏好/杠杆约束）。
本扫描在零幸存者偏差的申万官方行业指数上，用月频/季频检验这些因子。

因子: REV21(1月反转) / REV60(3月反转) / VOL60(低波) / MOM250(12月动量)
调仓: 月频 HOLD=21 / 季频 HOLD=63；TOP K=3,5；扣 0.1% 双边成本。
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sector_rotation_backtest import _tstat, COST_PCT  # noqa: E402
from sector_rotation_swindex_old import SW_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "sw_index_cache"
OUT = ROOT / "data" / "sw_factor_scan_report.json"
NEW_IN_2021 = {"煤炭", "石油石化", "环保", "美容护理"}
FACTORS = ("REV21", "REV60", "VOL60", "MOM250")
HOLDS = (21, 63)


def load_old_indexes():
    data = {}
    for cf in CACHE.glob("*.json"):
        name = SW_NAMES.get(cf.stem, cf.stem)
        if name in NEW_IN_2021:
            continue
        try:
            k = json.loads(cf.read_text(encoding="utf-8"))
            v = dict((str(d), float(c)) for d, c in k)
            if v and min(v) <= "2014-07-01":
                data[name] = v
        except Exception:
            pass
    return data


def main():
    data = load_old_indexes()
    sectors = sorted(data.keys())
    start = max(min(v) for v in data.values())
    dates = sorted({d for v in data.values() for d in v if d >= start})
    idx = {d: i for i, d in enumerate(dates)}
    arrs = {n: {idx[d]: c for d, c in v.items() if d in idx}
            for n, v in data.items()}
    print(f"行业指数 {len(sectors)} 个, {dates[0]} ~ {dates[-1]} "
          f"({len(dates)} 日, {len(dates)/244:.1f} 年)", flush=True)

    sec_ret = {n: {} for n in sectors}
    for t in range(1, len(dates)):
        for n in sectors:
            a = arrs[n]
            if t in a and (t - 1) in a and a[t - 1] > 0:
                sec_ret[n][t] = a[t] / a[t - 1] - 1

    def score(factor, t):
        out = {}
        for n in sectors:
            if factor == "VOL60":
                rs = [sec_ret[n].get(tt) for tt in range(t - 59, t + 1)]
                rs = [r for r in rs if r is not None]
                if len(rs) < 40:
                    continue
                out[n] = -st.pstdev(rs)          # 低波：波动取反
                continue
            lb = {"REV21": 21, "REV60": 60, "MOM250": 250}[factor]
            cum, ok = 1.0, True
            for tt in range(t - lb + 1, t + 1):
                r = sec_ret[n].get(tt)
                if r is None:
                    ok = False
                    break
                cum *= (1 + r)
            if not ok:
                continue
            perf = cum - 1
            out[n] = -perf if factor.startswith("REV") else perf
        return out

    def run(factor, K, hold, warm):
        prev, rets, bench, eq, peak, mdd = None, [], [], 1.0, 1.0, 0.0
        for t in range(warm, len(dates) - 1):
            if (t - warm) % hold != 0:
                continue
            sc = score(factor, t)
            if len(sc) < len(sectors) * 0.8:
                continue
            top = sorted(sc, key=lambda s: -sc[s])[:K]
            seg, bseg = [], []
            for tt in range(t + 1, min(t + hold + 1, len(dates))):
                rs = [sec_ret[s].get(tt) for s in top
                      if sec_ret[s].get(tt) is not None]
                if rs:
                    seg.append(st.mean(rs))
                a = [sec_ret[s].get(tt) for s in sectors
                     if sec_ret[s].get(tt) is not None]
                if a:
                    bseg.append(st.mean(a))
            if not seg:
                continue
            pr = 1.0
            for r in seg:
                pr *= (1 + r)
            pr -= 1
            cost = (COST_PCT / 100 * (1 - len(set(top) & set(prev)) / K)
                    if prev else 0.0)
            prev = top
            rets.append(((1 + pr) * (1 - cost) - 1) * 100)
            eq *= (1 + pr) * (1 - cost)
            peak = max(peak, eq)
            mdd = min(mdd, eq / peak - 1)
            if bseg:
                bench.append(st.mean(bseg) * hold * 100)
        return _tstat(rets), _tstat(bench), mdd * 100

    print(f"{'因子':>8}{'频率':>6}{'K':>3} | {'年化':>8}{'t':>7}{'回撤':>7} | "
          f"{'基准':>7} | {'超额':>8}", flush=True)
    results = []
    for factor in FACTORS:
        warm = {"REV21": 60, "REV60": 90, "MOM250": 300, "VOL60": 90}[factor]
        for hold in HOLDS:
            for K in (3, 5):
                s, b, mdd = run(factor, K, hold, warm)
                ann = s["mean"] * (252 / hold)
                bann = b["mean"] * (252 / hold)
                freq = "月频" if hold == 21 else "季频"
                results.append({"factor": factor, "hold": hold, "K": K,
                                "ann": round(ann, 1), "t": s["t"],
                                "bench": round(bann, 1),
                                "exc": round(ann - bann, 1),
                                "mdd": round(mdd, 1), "n": s["n"]})
                print(f"{factor:>8}{freq:>6}{K:>3} | {ann:>+7.1f}% {s['t']:>+6.2f} "
                      f"{mdd:>6.1f}% | {bann:>+6.1f}% | {ann - bann:>+7.1f}%",
                      flush=True)
    wins = sum(1 for r in results if r["t"] > 2)
    pos = sum(1 for r in results if r["exc"] > 0)
    print(f"\n判决: t>2 {wins}/{len(results)}; 超额为正 {pos}/{len(results)}; "
          f"中位超额 {st.median([r['exc'] for r in results]):+.1f}%/年", flush=True)
    best = max(results, key=lambda r: r["t"])
    print(f"最佳: {best['factor']} {'月' if best['hold']==21 else '季'}频 "
          f"K={best['K']} t={best['t']:+.2f} 超额{best['exc']:+.1f}%/年", flush=True)
    OUT.write_text(json.dumps({"start": dates[0], "end": dates[-1],
                               "n_sectors": len(sectors),
                               "results": results}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"已写 {OUT.name}", flush=True)


if __name__ == "__main__":
    main()
