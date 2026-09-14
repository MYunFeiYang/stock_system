#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""信号强度检验：强信号时入场，收益是否真的更好？

用户诉求"不一定要每天都买卖，但在值得的时候要买卖" —— 前提是系统能
区分"什么时候值得"。本脚本检验：动量信号的**强度**与后续收益是否单调。

方法（标准分档 IC 检验，零未来函数）：
  - 每个调仓日，对全池算 60 日动量，横截面 z-score 标准化
  - 按 z-score 分 5 档（D1 最弱 … D5 最强）
  - 统计各档买入后未来 HOLD(21/63) 日收益均值
  - 判据：①各档收益是否单调递增 ②D5 vs D1 的 t ③D5 vs D3(中档) 的 t
数据来源: 申万 27 老行业指数(零幸存者偏差) + 宽基 4 指数(21.6年)
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sector_rotation_backtest import _tstat  # noqa: E402
from sector_rotation_swindex_old import SW_NAMES  # noqa: E402
from style_rotation import fetch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SW_CACHE = ROOT / "data" / "sw_index_cache"
OUT = ROOT / "data" / "signal_strength_report.json"
NEW_IN_2021 = {"煤炭", "石油石化", "环保", "美容护理"}
LOOKBACK = 60
HOLDS = (21, 63)


def load_sw():
    data = {}
    for cf in SW_CACHE.glob("*.json"):
        name = SW_NAMES.get(cf.stem, cf.stem)
        if name in NEW_IN_2021:
            continue
        k = json.loads(cf.read_text(encoding="utf-8"))
        v = dict((str(d), float(c)) for d, c in k)
        if v and min(v) <= "2014-07-01":
            data[name] = v
    return data


def build(data):
    start = max(min(v) for v in data.values())
    dates = sorted({d for v in data.values() for d in v if d >= start})
    idx = {d: i for i, d in enumerate(dates)}
    arrs = {n: {idx[d]: c for d, c in v.items() if d in idx}
            for n, v in data.items()}
    sec = {n: {} for n in arrs}
    for t in range(1, len(dates)):
        for n in arrs:
            a = arrs[n]
            if t in a and (t - 1) in a and a[t - 1] > 0:
                sec[n][t] = a[t] / a[t - 1] - 1
    return dates, arrs, sec


def analyze(label, dates, sec, names):
    print(f"\n=== {label} ({len(names)} 标的, {dates[0]}~{dates[-1]}, "
          f"{len(dates)/244:.1f} 年) ===", flush=True)
    out_rows = []
    for hold in HOLDS:
        buckets = {i: [] for i in range(1, 6)}   # 1弱..5强
        for t in range(LOOKBACK, len(dates) - hold):
            if t % hold != 0:
                continue
            moms = {}
            for n in names:
                cum, ok = 1.0, True
                for tt in range(t - LOOKBACK + 1, t + 1):
                    r = sec[n].get(tt)
                    if r is None:
                        ok = False
                        break
                    cum *= (1 + r)
                if ok:
                    moms[n] = cum - 1
            if len(moms) < len(names) * 0.8:
                continue
            vals = list(moms.values())
            mu, sd = st.mean(vals), st.pstdev(vals)
            if sd <= 0:
                continue
            zs = {n: (m - mu) / sd for n, m in moms.items()}
            # 未来 hold 日收益
            fwd = {}
            for n in moms:
                cum, ok = 1.0, True
                for tt in range(t + 1, t + hold + 1):
                    r = sec[n].get(tt)
                    if r is None:
                        ok = False
                        break
                    cum *= (1 + r)
                if ok:
                    fwd[n] = cum - 1
            if not fwd:
                continue
            ranked = sorted(zs, key=lambda n: -zs[n])
            k = max(1, len(ranked) // 5)
            for i in range(5):
                grp = ranked[i * k:(i + 1) * k] if i < 4 else ranked[4 * k:]
                rs = [fwd[n] for n in grp if n in fwd]
                if rs:
                    # 档位: 索引0=最强 -> D5
                    buckets[5 - i].append(st.mean(rs) * 100)
        row = {"hold": hold}
        print(f"  HOLD={hold}日:")
        for i in range(1, 6):
            s = _tstat(buckets[i])
            ann = s["mean"] * (252 / hold)
            row[f"D{i}_n"] = s["n"]
            row[f"D{i}_ann_pct"] = round(ann, 1)
            row[f"D{i}_t"] = s["t"]
            print(f"    D{i}{'(最强)' if i==5 else ''}: 年化{ann:>+7.1f}% "
                  f"t={s['t']:>+5.2f} n={s['n']}")
        # 单调性
        anns = [row[f"D{i}_ann_pct"] for i in range(1, 6)]
        mono = all(anns[i] <= anns[i + 1] for i in range(4))
        # D5 vs D1
        d5, d1 = buckets[5], buckets[1]
        paired = _tstat([a - b for a, b in zip(d5, d1)])
        print(f"    单调性: {'✓ 递增' if mono else '✗ 非单调'}; "
              f"D5-D1 配对 t={paired['t']:+.2f} 均值差{paired['mean']:+.3f}%")
        row["monotonic"] = mono
        row["d5_minus_d1_t"] = paired["t"]
        row["d5_minus_d1_mean_pct"] = paired["mean"]
        out_rows.append(row)
    return out_rows


def main():
    res = {}
    sw = load_sw()
    dates, arrs, sec = build(sw)
    res["sw_industry"] = analyze("申万27行业指数(零偏差)", dates, sec,
                                 sorted(sec.keys()))
    style = {n: dict(fetch(s)) for s, n in
             {"sh000300": "沪深300", "sh000016": "上证50",
              "sh000905": "中证500", "sh000015": "上证红利"}.items()}
    d2, a2, sec2 = build(style)
    res["style_4index"] = analyze("宽基4指数(21.6年)", d2, sec2,
                                  sorted(sec2.keys()))
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n已写 {OUT.name}", flush=True)


if __name__ == "__main__":
    main()
