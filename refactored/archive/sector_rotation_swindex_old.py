#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""27 个老行业申万指数（2021 前存在）· 2014 起 12 年零幸存者偏差验证。

直接读 sw_index_cache 缓存（sector_rotation_swindex.py 已拉取），
不 import akshare。剔除 2021 版新增行业（煤炭/石油石化/环保/美容护理），
用其余老行业指数从共同覆盖段起点回测——申万指数含历史全部成分，
零幸存者偏差，是行业轮动策略的最终裁判。
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sector_rotation_backtest import _tstat, COST_PCT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "sw_index_cache"
OUT = ROOT / "data" / "sector_rotation_swindex_old27_report.json"
LOOKBACKS = (20, 40, 60, 120)
TOPS = (2, 3, 5)
HOLD = 5
NEW_IN_2021 = {"煤炭", "石油石化", "环保", "美容护理"}


def main():
    data = {}
    for cf in CACHE.glob("*.json"):
        code = cf.stem
        try:
            k = json.loads(cf.read_text(encoding="utf-8"))
            name = SW_NAMES.get(code, code)
            data[name] = dict((str(d), float(c)) for d, c in k)
        except Exception as e:
            print(f"skip {code}: {e}")
    old = {n: v for n, v in data.items()
           if n not in NEW_IN_2021 and min(v) <= "2014-07-01"}
    print(f"老行业指数: {len(old)}/{len(data)} 个", flush=True)
    sectors = sorted(old.keys())
    start = max(min(v) for v in old.values())
    dates = sorted({d for v in old.values() if True for d in v if d >= start})
    idx = {d: i for i, d in enumerate(dates)}
    arrs = {n: {idx[d]: c for d, c in v.items() if d in idx}
            for n, v in old.items()}
    years = len(dates) / 244
    print(f"覆盖: {dates[0]} ~ {dates[-1]} ({len(dates)} 日, {years:.1f} 年)",
          flush=True)

    sec_ret = {n: {} for n in sectors}
    for t in range(1, len(dates)):
        for n in sectors:
            a = arrs[n]
            if t in a and (t - 1) in a and a[t - 1] > 0:
                sec_ret[n][t] = a[t] / a[t - 1] - 1

    def run(lb, K):
        prev, rets, bench, eq, peak, mdd = None, [], [], 1.0, 1.0, 0.0
        for t in range(lb, len(dates) - 1):
            if (t - lb) % HOLD != 0:
                continue
            perf = {}
            for n in sectors:
                cum, ok = 1.0, True
                for tt in range(t - lb + 1, t + 1):
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
                bench.append(st.mean(bseg) * HOLD * 100)
        return _tstat(rets), _tstat(bench), mdd * 100

    print(f"{'lb':>4}{'top':>4} | {'策略年化':>8}{'t':>7}{'回撤':>7} | "
          f"{'基准年化':>8} | {'超额':>8}", flush=True)
    results = []
    for lb in LOOKBACKS:
        for K in TOPS:
            s, b, mdd = run(lb, K)
            ann = s["mean"] * (252 / HOLD)
            bann = b["mean"] * (252 / HOLD)
            results.append({"lb": lb, "top": K, "ann": round(ann, 1),
                            "t": s["t"], "bench": round(bann, 1),
                            "exc": round(ann - bann, 1), "mdd": round(mdd, 1),
                            "n": s["n"]})
            print(f"{lb:>4}{K:>4} | {ann:>+7.1f}% {s['t']:>+6.2f} {mdd:>6.1f}% "
                  f"| {bann:>+7.1f}% | {ann - bann:>+7.1f}%", flush=True)
    wins = sum(1 for r in results if r["t"] > 2)
    pos = sum(1 for r in results if r["exc"] > 0)
    med = st.median([r["exc"] for r in results])
    print(f"\n判决({len(sectors)}行业指数, {years:.1f}年): t>2 {wins}/12; "
          f"超额为正 {pos}/12; 中位超额 {med:+.1f}%/年", flush=True)
    OUT.write_text(json.dumps(
        {"start": dates[0], "end": dates[-1], "n_sectors": len(sectors),
         "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写 {OUT.name}", flush=True)


SW_NAMES = {
    "801010": "农林牧渔", "801030": "基础化工", "801040": "钢铁",
    "801050": "有色金属", "801080": "电子", "801110": "家用电器",
    "801120": "食品饮料", "801130": "纺织服饰", "801140": "轻工制造",
    "801150": "医药生物", "801160": "公用事业", "801170": "交通运输",
    "801180": "房地产", "801200": "商贸零售", "801210": "社会服务",
    "801230": "综合", "801710": "建筑材料", "801720": "建筑装饰",
    "801730": "电力设备", "801740": "国防军工", "801750": "计算机",
    "801760": "传媒", "801770": "通信", "801780": "银行",
    "801790": "非银金融", "801880": "汽车", "801890": "机械设备",
    "801950": "煤炭", "801960": "石油石化", "801970": "环保",
    "801980": "美容护理",
}


if __name__ == "__main__":
    main()
