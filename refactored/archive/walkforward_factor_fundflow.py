"""主力资金流因子 walk-forward 验证（② 第一步）。

数据源: eastmoney push2his 个股资金流日线（直连 HTTP，免 akshare）。
口径: 每只股票取「主力净流入率(%)」(f53) 作为因子，按 pred_day 与
      walkforward_raw.json 的 ret_next(隔日收益) 做 point-in-time join
      （因子在 pred_day 收盘已知，预测其后隔日收益，无未来函数）。

测试:
  - IC = Pearson(因子, ret_next) 全样本
  - 每日按因子排序，前 1/10 等权(做多) 的隔日组合 t
  - 多空(前1/10 − 后1/10) 隔日组合 t
  - 二值: 因子>0(主力净流入) 当作买入规则的 t
  - 复合: 因子>0 且 regime==range 的 t（叠加现有最佳口袋）
"""
from __future__ import annotations

import json
import math
import statistics as st
import urllib.request
from pathlib import Path

RAW = Path("data/walkforward_raw.json")
OUT = Path("data/factor_fundflow_report.json")


def _tstat(xs: list) -> dict:
    n = len(xs)
    if n < 2:
        return {"n": n, "mean": (xs[0] if n == 1 else 0.0), "sd": 0.0,
                "t": 0.0, "significant": False}
    m = st.mean(xs)
    sd = st.stdev(xs)
    se = sd / math.sqrt(n) if sd else 0.0
    t = (m / se) if se else 0.0
    return {"n": n, "mean": round(m, 4), "sd": round(sd, 4),
            "t": round(t, 2), "significant": abs(t) > 2.0}


def _by_day_series(rows: list) -> list:
    bucket: dict[str, list] = {}
    for r in rows:
        v = r.get("ret_next")
        if v is None:
            continue
        bucket.setdefault(r["pred_day"], []).append(v)
    return [st.mean(v) for _, v in sorted(bucket.items()) if v]


def _pearson(xs: list, ys: list) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = st.mean(xs), st.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return round(cov / (sx * sy), 3) if sx and sy else 0.0


def fetch_fund_flow(code: str, days: int = 400) -> dict:
    """返回 {day: 主力净流入率%}；secid: 沪(6*)→1. 深→0."""
    import time
    secid = ("1." if code.startswith("6") else "0.") + code
    url = (f"https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
           f"?lmt={days}&klt=101&secid={secid}"
           f"&fields1=f1,f2,f3,f7"
           f"&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65"
           f"&ut=b2884a393a59ad64002292a3e90d46a5")
    last = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "replace")
            obj = json.loads(raw)
            kls = (obj.get("data") or {}).get("klines") or []
            out = {}
            for row in kls:
                f = row.split(",")
                if len(f) > 2 and f[0]:
                    try:
                        out[f[0]] = float(f[2])   # f53 主力净流入率%
                    except ValueError:
                        pass
            return out
        except Exception as e:
            last = e
            time.sleep(0.5 * (attempt + 1))
    print(f"  ⚠️ {code} 资金流抓取失败: {last}")
    return {}


def main():
    blob = json.load(open(RAW, encoding="utf-8"))
    rows = blob["rows"]
    symbols = sorted({r["symbol"] for r in rows})
    print(f"【主力资金流因子验证】股票池 {len(symbols)} 只 | raw 区间 {blob['range']['start']}~{blob['range']['end']}")

    # 1) 拉每只股票资金流日线
    flow: dict[str, dict] = {}
    import time
    for sym in symbols:
        ff = fetch_fund_flow(sym)
        time.sleep(0.15)
        if ff:
            flow[sym] = ff
            print(f"  {sym}: 资金流 {len(ff)} 日 ({min(ff)}~{max(ff)})")
        else:
            print(f"  {sym}: 无资金流")

    # 2) point-in-time join
    joined = []
    for r in rows:
        ff = flow.get(r["symbol"])
        if not ff:
            continue
        fp = ff.get(r["pred_day"])
        if fp is None:
            continue
        joined.append({**r, "flow_pct": fp})
    print(f"\n  join 成功 {len(joined)} 条（因子×隔日收益，覆盖 {len({j['pred_day'] for j in joined})} 交易日）")

    if not joined:
        raise SystemExit("无 join 样本")

    # 3) 统计
    xs = [j["flow_pct"] for j in joined]
    ys = [j["ret_next"] for j in joined]
    ic = _pearson(xs, ys)
    print(f"\n  IC = Pearson(主力净流入率, 隔日收益) = {ic:+.3f}（n={len(joined)}）")

    # 每日排序做多前1/10 / 多空
    by_day: dict[str, list] = {}
    for j in joined:
        by_day.setdefault(j["pred_day"], []).append(j)
    long_top, long_short = [], []
    pos_only, comb = [], []
    for day, grp in sorted(by_day.items()):
        grp = sorted(grp, key=lambda x: x["flow_pct"])
        k = max(1, len(grp) // 10)
        top = [x["ret_next"] for x in grp[-k:]]
        bot = [x["ret_next"] for x in grp[:k]]
        long_top.append(st.mean(top))
        long_short.append(st.mean(top) - st.mean(bot))
        pos_only += [x for x in grp if x["flow_pct"] > 0]
        comb += [x for x in grp if x["flow_pct"] > 0 and x["regime"] == "range"]

    def _show(label, dayxs, ntrades=None):
        s = _tstat(dayxs)
        ws = round(100.0 * sum(1 for x in dayxs if x > 0) / len(dayxs), 1)
        print(f"  {label:<26} 日数={s['n']:>3} 均值/日={s['mean']:+.3f}% "
              f"t={s['t']:+.2f} 胜日={ws}% {'★显著' if s['significant'] else ''}"
              + (f" (笔数={ntrades})" if ntrades else ""))

    print("\n=== 资金流因子 walk-forward（按天主判据）===")
    _show("做多前1/10(因子高)", long_top)
    _show("多空(前1/10−后1/10)", long_short)
    _show("二值: 主力净流入>0", _by_day_series(pos_only), len(pos_only))
    _show("复合: 净流入>0 & range", _by_day_series(comb), len(comb))

    # 4) 落盘
    out = {
        "ic": ic,
        "n_joined": len(joined),
        "flow_window": {"start": min(min(f) for f in flow.values()),
                        "end": max(max(f) for f in flow.values())},
        "long_top": _tstat(long_top),
        "long_short": _tstat(long_short),
        "binary_inflow": _tstat(_by_day_series(pos_only)) | {"n_trades": len(pos_only)},
        "combo_inflow_range": _tstat(_by_day_series(comb)) | {"n_trades": len(comb)},
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n💾 {OUT}")

    # 5) 结论
    best = max(out["long_top"]["t"], out["long_short"]["t"],
               out["binary_inflow"]["t"], out["combo_inflow_range"]["t"])
    print("\n=== 结论 ===")
    if best > 2.0:
        print(f"  ✅ 资金流因子有显著正 edge（最佳 t={best:+.2f}），可进 ② 合成。")
    elif best < -2.0:
        print(f"  ❌ 资金流因子显著负（最佳 t={best:+.2f}）——此维度也无 edge。")
    else:
        print(f"  ⚠️ 资金流因子不显著（|t|<=2，最佳 t={best:+.2f}）——"
              "单因子无 edge；需换维度(估值/成长/北向)或合成多因子。")


if __name__ == "__main__":
    main()
