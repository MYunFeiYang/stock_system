#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""风格轮动 · 老 4 指数长历史版（沪深300/上证50/中证500/上证红利, 2007 起 ~19 年）。

剔除中证1000(2014起)与创业板指(2010起)以换取更长样本——零花费下提升
统计功效的唯一手段是拉长历史而非换数据源。sina datalen=6000 上限 5986 根。
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sector_rotation_backtest import _tstat, COST_PCT  # noqa: E402
from style_rotation import fetch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "style_rotation_old4_19y_report.json"
SYM = {"sh000300": "沪深300", "sh000016": "上证50",
       "sh000905": "中证500", "sh000015": "上证红利"}
FACTORS = ("MOM60", "MOM120", "MOM250", "VOL60")


def main():
    data = {n: dict(fetch(s)) for s, n in SYM.items()}
    start = max(min(v) for v in data.values())
    dates = sorted({d for v in data.values() for d in v if d >= start})
    idx = {d: i for i, d in enumerate(dates)}
    arrs = {n: {idx[d]: c for d, c in v.items() if d in idx}
            for n, v in data.items()}
    names = sorted(arrs)
    print(f"标的 {names}")
    print(f"覆盖 {dates[0]}~{dates[-1]} ({len(dates)} 日, "
          f"{len(dates)/244:.1f} 年)", flush=True)

    ret = {n: {} for n in names}
    for t in range(1, len(dates)):
        for n in names:
            a = arrs[n]
            if t in a and (t - 1) in a and a[t - 1] > 0:
                ret[n][t] = a[t] / a[t - 1] - 1

    def score(f, t):
        out = {}
        for n in names:
            if f == "VOL60":
                rs = [ret[n].get(tt) for tt in range(t - 59, t + 1)]
                rs = [r for r in rs if r is not None]
                if len(rs) < 40:
                    continue
                out[n] = -st.pstdev(rs)
                continue
            lb = {"MOM60": 60, "MOM120": 120, "MOM250": 250}[f]
            cum, ok = 1.0, True
            for tt in range(t - lb + 1, t + 1):
                r = ret[n].get(tt)
                if r is None:
                    ok = False
                    break
                cum *= (1 + r)
            if ok:
                out[n] = cum - 1
        return out

    def run(f, K, hold, warm):
        prev, rets, bench, eq, peak, mdd = None, [], [], 1.0, 1.0, 0.0
        for t in range(warm, len(dates) - 1):
            if (t - warm) % hold != 0:
                continue
            sc = score(f, t)
            if len(sc) < 3:
                continue
            top = sorted(sc, key=lambda s: -sc[s])[:K]
            seg, bseg = [], []
            for tt in range(t + 1, min(t + hold + 1, len(dates))):
                rs = [ret[s].get(tt) for s in top
                      if ret[s].get(tt) is not None]
                if rs:
                    seg.append(st.mean(rs))
                a = [ret[s].get(tt) for s in names
                     if ret[s].get(tt) is not None]
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
        # 超额序列（同长度配对）——绝对收益 t 不能论证策略，必须检验超额
        ex_t = _tstat([a - b for a, b in zip(rets, bench)])
        return _tstat(rets), _tstat(bench), mdd * 100, ex_t

    print(f"{'因子':>8}{'频率':>6}{'K':>3} | {'年化':>8}{'绝对t':>7}{'回撤':>7} | "
          f"{'基准':>7} | {'超额':>8}{'超额t':>8}", flush=True)
    results = []
    for f in FACTORS:
        warm = {"MOM60": 90, "MOM120": 150, "MOM250": 300, "VOL60": 90}[f]
        for hold in (21, 63):
            for K in (1, 2):
                s, b, mdd, ex = run(f, K, hold, warm)
                ann = s["mean"] * (252 / hold)
                bann = b["mean"] * (252 / hold)
                results.append({"f": f, "hold": hold, "K": K,
                                "ann": round(ann, 1), "t": s["t"],
                                "bench": round(bann, 1),
                                "exc": round(ann - bann, 1),
                                "exc_t": ex["t"], "exc_n": ex["n"],
                                "mdd": round(mdd, 1), "n": s["n"]})
                print(f"{f:>8}{'月' if hold == 21 else '季'}频{K:>3} | "
                      f"{ann:>+7.1f}% {s['t']:>+6.2f} {mdd:>6.1f}% | "
                      f"{bann:>+6.1f}% | {ann - bann:>+7.1f}% {ex['t']:>+7.2f}",
                      flush=True)
    wins = sum(1 for r in results if r["t"] > 2)
    exwins = sum(1 for r in results if r["exc_t"] > 2)
    pos = sum(1 for r in results if r["exc"] > 0)
    med = st.median([r["exc"] for r in results])
    print(f"\n判决({len(dates)/244:.0f}年): 绝对收益 t>2 {wins}/{len(results)}; "
          f"**超额 t>2 {exwins}/{len(results)}**; 超额为正 {pos}/{len(results)}; "
          f"中位超额 {med:+.1f}%/年", flush=True)
    best = max(results, key=lambda r: r["t"])
    print(f"最佳: {best['f']} {'月' if best['hold'] == 21 else '季'}频 "
          f"K={best['K']} t={best['t']:+.2f} 超额{best['exc']:+.1f}%/年 "
          f"回撤{best['mdd']:.1f}%", flush=True)
    OUT.write_text(json.dumps({"start": dates[0], "end": dates[-1],
                               "symbols": names, "results": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写 {OUT.name}", flush=True)


if __name__ == "__main__":
    main()
