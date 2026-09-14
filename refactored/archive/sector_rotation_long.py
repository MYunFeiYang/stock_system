#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""行业动量轮动 · 12 年跨牛熊回测（sina 3000 根，2014-05 ~ 2026-09）。

目的：3 年样本只覆盖一个牛段，"市况依赖"只有半截证据。本回测把策略放进
2015 股灾、2018 熊市、2019-2021 结构牛、2022-2024 熊、2024-2026 牛全面检验。

口径：与 sector_rotation.py 相同（行业等权、周频、TOP-K、扣 0.1% ETF 成本、
零未来函数）。局限：sina 不复权，绝对收益约低估股息率(1~2%/年)，
除权跳变在行业等权中被 5~6 只成分稀释，对行业间相对排序影响有限——
而排序正是行业动量的核心。
"""
import json
import math
import statistics as st
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_relative_strength import load_pool  # noqa: E402
from sector_rotation_backtest import _tstat, COST_PCT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "history_cache_3000"
CACHE.mkdir(parents=True, exist_ok=True)
LOOKBACKS = (20, 40, 60, 120)
TOPS = (2, 3, 5)
HOLD = 5
ERAS = [  # (名称, 起日, 止日) — 按上证指数经典牛熊划分
    ("2014-2015牛", "2014-05-01", "2015-06-12"),
    ("2015股灾", "2015-06-15", "2016-01-31"),
    ("2016-2018熊", "2016-02-01", "2018-12-31"),
    ("2019-2021牛", "2019-01-01", "2021-12-31"),
    ("2022-2024熊", "2022-01-01", "2024-09-23"),
    ("2024-2026牛", "2024-09-24", "2026-09-30"),
]


def _sina_symbol(code):
    return ("sh" if code.startswith("6") else "sz") + code


def get_history_long(code):
    cf = CACHE / f"{code}.json"
    if cf.exists():
        try:
            return json.loads(cf.read_text(encoding="utf-8"))
        except Exception:
            pass
    u = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
         f"CN_MarketData.getKLineData?symbol={_sina_symbol(code)}"
         "&scale=240&ma=no&datalen=3000")
    try:
        raw = urllib.request.urlopen(urllib.request.Request(
            u, headers={"User-Agent": "Mozilla/5.0",
                        "Referer": "https://finance.sina.com.cn"}),
            timeout=25).read().decode()
        d = json.loads(raw)
        out = [(r["day"], float(r["close"])) for r in d if r.get("close")]
        cf.write_text(json.dumps(out), encoding="utf-8")
        return out
    except Exception:
        return []


def main():
    stocks = load_pool()
    print(f"[1/3] 拉取/读取 3000 根长历史...")
    hist = {}
    for s in stocks:
        h = get_history_long(s["symbol"])
        if h:
            hist[s["symbol"]] = h
    print(f"      可用: {len(hist)}/{len(stocks)} 只")

    sec_of = {s["symbol"]: s.get("sector", "未分类") for s in stocks}
    valid = {c: dict(h) for c, h in hist.items() if len(h) > 130}
    dates = sorted({d for v in valid.values() for d in v})
    idx = {d: i for i, d in enumerate(dates)}
    arrs = {c: {idx[d]: p for d, p in v.items()} for c, v in valid.items()}
    sectors = sorted({sec_of[c] for c in valid})
    print(f"[2/3] 行业收益: {len(sectors)} 行业 × {len(dates)} 日 "
          f"({dates[0]} ~ {dates[-1]})")
    sec_ret = {sec: {} for sec in sectors}
    for t in range(1, len(dates)):
        by = {}
        for c, a in arrs.items():
            if t in a and (t - 1) in a and a[t - 1] > 0:
                by.setdefault(sec_of[c], []).append(a[t] / a[t - 1] - 1)
        for sec, rs in by.items():
            sec_ret[sec][t] = st.mean(rs)

    def run(lookback, K, t_from, t_to):
        prev, rets, eq, peak, mdd = None, [], 1.0, 1.0, 0.0
        for t in range(max(lookback, t_from), t_to):
            if (t - lookback) % HOLD != 0:
                continue
            perf = {}
            for sec in sectors:
                cum, ok = 1.0, True
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
            top = sorted(perf, key=lambda s: -perf[s])[:K]
            seg = []
            for tt in range(t + 1, min(t + HOLD + 1, len(dates))):
                rs = [sec_ret[s].get(tt) for s in top
                      if sec_ret[s].get(tt) is not None]
                if rs:
                    seg.append(st.mean(rs))
            if not seg:
                continue
            pr = 1.0
            for r in seg:
                pr *= (1 + r)
            pr -= 1
            cost = (COST_PCT / 100 * (1 - len(set(top) & set(prev)) / K)
                    if prev else 0.0)
            prev = top
            net = (1 + pr) * (1 - cost) - 1
            rets.append(net * 100)
            eq *= (1 + net)
            peak = max(peak, eq)
            mdd = min(mdd, eq / peak - 1)
        return _tstat(rets), mdd * 100

    print("[3/3] 12 组参数全平面 × 牛熊分段:")
    print(f"\n{'lb':>4}{'top':>4} | {'全样本年化':>9}{'t':>6}{'回撤':>7} | "
          + " | ".join(n[:9] for n, _, _ in ERAS))
    all_res = []
    for lookback in LOOKBACKS:
        for K in TOPS:
            s_full, mdd = run(lookback, K, 0, len(dates) - 1)
            ann_full = s_full["mean"] * (252 / HOLD)
            cells = []
            for name, d0, d1 in ERAS:
                t0, t1 = idx.get(d0, 0), idx.get(d1, len(dates) - 1)
                s_e, _ = run(lookback, K, t0, t1)
                cells.append(s_e["mean"] * (252 / HOLD))
            all_res.append({"lookback": lookback, "top": K,
                            "full_annual": round(ann_full, 1), "t": s_full["t"],
                            "mdd": round(mdd, 1),
                            "eras": {n: round(c, 1) for (n, _, _), c
                                     in zip(ERAS, cells)}})
            print(f"{lookback:>4}{K:>4} | {ann_full:>+8.1f}% {s_full['t']:>+6.2f} "
                  f"{mdd:>6.1f}% | " + " | ".join(f"{c:>+7.1f}" for c in cells))

    # 汇总：牛段 vs 熊段
    bull_idx = [0, 3, 5]
    bear_idx = [1, 2, 4]
    bull_pos = sum(1 for r in all_res
                   if sum(r["eras"][ERAS[i][0]] for i in bull_idx) > 0)
    bear_pos = sum(1 for r in all_res
                   if sum(r["eras"][ERAS[i][0]] for i in bear_idx) > 0)
    print(f"\n牛段(2015/2019/2024)为正: {bull_pos}/{len(all_res)} 组; "
          f"熊段(2015H2/2016-18/2022-24)为正: {bear_pos}/{len(all_res)} 组")
    (ROOT / "data" / "sector_rotation_long_report.json").write_text(
        json.dumps({"eras": [n for n, _, _ in ERAS], "results": all_res},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("已写 data/sector_rotation_long_report.json")


if __name__ == "__main__":
    main()
