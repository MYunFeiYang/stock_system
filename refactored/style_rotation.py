#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""宽基/风格指数轮动（零花费、零幸存者偏差、ETF 可执行）。

标的（均可用对应 ETF 实盘买入）：
  sh000300 沪深300 / sh000016 上证50 / sh000905 中证500 /
  sh000852 中证1000 / sz399006 创业板指 / sh000015 上证红利
覆盖规模（大盘/中盘/小盘）+ 风格（成长/红利）维度，官方编制，
无幸存者偏差，数据全部免费。

因子: MOM60 / MOM120 / MOM250 / VOL60(低波)
调仓: 月频(21) / 季频(63)；TOP1 / TOP2；扣 0.1% 双边成本。
"""
import json
import statistics as st
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sector_rotation_backtest import _tstat, COST_PCT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "style_index_cache_6000"
CACHE.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "data" / "style_rotation_long_report.json"
SYMBOLS = {
    "sh000300": "沪深300", "sh000016": "上证50", "sh000905": "中证500",
    "sh000852": "中证1000", "sz399006": "创业板指", "sh000015": "上证红利",
}
ETF = {"沪深300": "510300", "上证50": "510050", "中证500": "510500",
       "中证1000": "512100", "创业板指": "159915", "上证红利": "510880"}
FACTORS = ("MOM60", "MOM120", "MOM250", "VOL60")
HOLDS = (21, 63)


def fetch(sym):
    cf = CACHE / f"{sym}_6000.json"
    if cf.exists():
        try:
            return json.loads(cf.read_text(encoding="utf-8"))
        except Exception:
            pass
    u = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
         f"CN_MarketData.getKLineData?symbol={sym}&scale=240&ma=no&datalen=6000")
    raw = urllib.request.urlopen(urllib.request.Request(
        u, headers={"User-Agent": "Mozilla/5.0",
                    "Referer": "https://finance.sina.com.cn"}),
        timeout=25).read().decode()
    d = json.loads(raw)
    out = [(r["day"], float(r["close"])) for r in d if r.get("close")]
    cf.write_text(json.dumps(out), encoding="utf-8")
    return out


def main():
    data = {}
    for sym, name in SYMBOLS.items():
        data[name] = dict(fetch(sym))
    start = max(min(v) for v in data.values())
    dates = sorted({d for v in data.values() for d in v if d >= start})
    idx = {d: i for i, d in enumerate(dates)}
    arrs = {n: {idx[d]: c for d, c in v.items() if d in idx}
            for n, v in data.items()}
    names = sorted(arrs.keys())
    print(f"标的 {len(names)} 个: {names}")
    print(f"覆盖: {dates[0]} ~ {dates[-1]} ({len(dates)} 日, "
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
            if len(sc) < 4:
                continue
            top = sorted(sc, key=lambda s: -sc[s])[:K]
            seg, bseg = [], []
            for tt in range(t + 1, min(t + hold + 1, len(dates))):
                rs = [ret[s].get(tt) for s in top if ret[s].get(tt) is not None]
                if rs:
                    seg.append(st.mean(rs))
                a = [ret[s].get(tt) for s in names if ret[s].get(tt) is not None]
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

    print(f"\n{'因子':>8}{'频率':>6}{'K':>3} | {'年化':>8}{'t':>7}{'回撤':>7} | "
          f"{'基准':>7} | {'超额':>8}", flush=True)
    results = []
    for f in FACTORS:
        warm = {"MOM60": 90, "MOM120": 150, "MOM250": 300, "VOL60": 90}[f]
        for hold in HOLDS:
            for K in (1, 2):
                s, b, mdd = run(f, K, hold, warm)
                ann = s["mean"] * (252 / hold)
                bann = b["mean"] * (252 / hold)
                freq = "月频" if hold == 21 else "季频"
                results.append({"factor": f, "hold": hold, "K": K,
                                "ann": round(ann, 1), "t": s["t"],
                                "bench": round(bann, 1),
                                "exc": round(ann - bann, 1),
                                "mdd": round(mdd, 1), "n": s["n"]})
                print(f"{f:>8}{freq:>6}{K:>3} | {ann:>+7.1f}% {s['t']:>+6.2f} "
                      f"{mdd:>6.1f}% | {bann:>+6.1f}% | {ann - bann:>+7.1f}%",
                      flush=True)
    wins = sum(1 for r in results if r["t"] > 2)
    pos = sum(1 for r in results if r["exc"] > 0)
    print(f"\n判决: t>2 {wins}/{len(results)}; 超额为正 {pos}/{len(results)}; "
          f"中位超额 {st.median([r['exc'] for r in results]):+.1f}%/年", flush=True)
    best = max(results, key=lambda r: r["t"])
    print(f"最佳: {best['factor']} {'月' if best['hold']==21 else '季'}频 "
          f"K={best['K']} t={best['t']:+.2f} 超额{best['exc']:+.1f}%/年 "
          f"回撤{best['mdd']:.1f}%", flush=True)
    OUT.write_text(json.dumps({"start": dates[0], "end": dates[-1],
                               "symbols": names, "etf": ETF,
                               "results": results}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"已写 {OUT.name}", flush=True)


if __name__ == "__main__":
    main()
