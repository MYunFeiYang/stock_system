#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""行业动量轮动 · 申万官方指数版（黄金标准验证）。

与 sector_rotation_long.py 的区别：不用"当前龙头股合成行业"，而是直接用
申万一级行业指数（ak.index_hist_sw，1999 年起）——指数含历史全部成分
（含后来退市/调出的股票），**零幸存者偏差**，且与实盘行业 ETF 的跟踪
标的一致。这是行业轮动策略的最终裁判。

口径：周频、lookback × TOP-K 12 组参数、扣 0.1% 双边成本、零未来函数。
基准：31 行业等权。
"""
import json
import math
import statistics as st
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sector_rotation_backtest import _tstat, COST_PCT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "sw_index_cache"
CACHE.mkdir(parents=True, exist_ok=True)
LOOKBACKS = (20, 40, 60, 120)
TOPS = (2, 3, 5)
HOLD = 5


def fetch_sw_index(code: str):
    cf = CACHE / f"{code}.json"
    if cf.exists():
        try:
            return json.loads(cf.read_text(encoding="utf-8"))
        except Exception:
            pass
    import akshare as ak
    df = ak.index_hist_sw(symbol=code, period="day")
    out = [[str(r["日期"]), float(r["收盘"])] for _, r in df.iterrows()
           if r["收盘"] and r["收盘"] > 0]
    cf.write_text(json.dumps(out), encoding="utf-8")
    return out


def main():
    import akshare as ak
    print("[1/3] 拉取申万一级行业指数...")
    info = ak.sw_index_first_info()
    codes = {}
    for _, r in info.iterrows():
        codes[str(r["行业代码"])[:6]] = r["行业名称"]
    print(f"      一级行业: {len(codes)} 个")

    data = {}
    for code, name in codes.items():
        try:
            k = fetch_sw_index(code)
            if len(k) > 200:
                data[name] = dict((d, c) for d, c in k)
        except Exception as e:
            print(f"      {code} {name} 失败: {e}")
    print(f"      可用: {len(data)}/{len(codes)}")

    sectors = sorted(data.keys())
    dates = sorted({d for v in data.values() for d in v})
    # 只用所有行业共同覆盖的日期段
    starts = [min(v) for v in data.values()]
    common_start = max(starts)
    dates = [d for d in dates if d >= common_start]
    idx = {d: i for i, d in enumerate(dates)}
    arrs = {n: {idx[d]: c for d, c in v.items() if d in idx}
            for n, v in data.items()}
    print(f"[2/3] 共同覆盖段: {dates[0]} ~ {dates[-1]} ({len(dates)} 日, "
          f"{len(dates)/244:.1f} 年)")

    sec_ret = {n: {} for n in sectors}
    for t in range(1, len(dates)):
        for n in sectors:
            a = arrs[n]
            if t in a and (t - 1) in a and a[t - 1] > 0:
                sec_ret[n][t] = a[t] / a[t - 1] - 1

    def run(lookback, K, t_from, t_to):
        prev, rets, bench, eq, peak, mdd = None, [], [], 1.0, 1.0, 0.0
        for t in range(max(lookback, t_from), t_to):
            if (t - lookback) % HOLD != 0:
                continue
            perf = {}
            for n in sectors:
                cum, ok = 1.0, True
                for tt in range(t - lookback + 1, t + 1):
                    r = sec_ret[n].get(tt)
                    if r is None:
                        ok = False
                        break
                    cum *= (1 + r)
                if ok:
                    perf[n] = cum - 1
            if len(perf) < len(sectors) * 0.8:
                continue
            top = sorted(perf, key=lambda s: -perf[s])[:K]
            seg, bseg = [], []
            for tt in range(t + 1, min(t + HOLD + 1, len(dates))):
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
                bench.append((st.mean(bseg) * HOLD) * 100)
        return _tstat(rets), _tstat(bench), mdd * 100

    print("[3/3] 12 组参数（申万指数口径，零幸存者偏差）:")
    print(f"{'lb':>4}{'top':>4} | {'策略年化':>8}{'t':>7}{'回撤':>7} | "
          f"{'基准年化':>8} | {'超额':>8}")
    results = []
    for lb in LOOKBACKS:
        for K in TOPS:
            s, b, mdd = run(lb, K, 0, len(dates) - 1)
            ann = s["mean"] * (252 / HOLD)
            bann = b["mean"] * (252 / HOLD)
            results.append({"lookback": lb, "top": K,
                            "annual_pct": round(ann, 1), "t": s["t"],
                            "bench_annual_pct": round(bann, 1),
                            "excess_annual_pct": round(ann - bann, 1),
                            "mdd_pct": round(mdd, 1), "n": s["n"]})
            print(f"{lb:>4}{K:>4} | {ann:>+7.1f}% {s['t']:>+6.2f} {mdd:>6.1f}% | "
                  f"{bann:>+7.1f}% | {ann - bann:>+7.1f}%")
    wins = sum(1 for r in results if r["t"] > 2)
    pos_ex = sum(1 for r in results if r["excess_annual_pct"] > 0)
    print(f"\n判决: t>2 的 {wins}/12; 超额为正 {pos_ex}/12; "
          f"中位超额 {st.median([r['excess_annual_pct'] for r in results]):+.1f}%/年")
    (ROOT / "data" / "sector_rotation_swindex_report.json").write_text(
        json.dumps({"start": dates[0], "end": dates[-1],
                    "n_days": len(dates), "results": results},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("已写 data/sector_rotation_swindex_report.json")


if __name__ == "__main__":
    main()
