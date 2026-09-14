"""walk-forward 均值回归买点验证（离线，依赖 data/walkforward_raw.json）。

用法:
  python3 refactored/walkforward_meanreversion.py

对 raw 逐笔记录施加多种「均值回归买点」规则，计算其隔日买入组合的
walk-forward 按天主判据 t 值，与原始公式 / 闸门-range 基线对比。
所有规则仅用「决策时已可得」的价量特征 → 隔日收益，零未来函数。
"""
from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

RAW = Path("data/walkforward_raw.json")
BUY_SIGNALS = ("买入", "强烈买入")


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


def _by_day_series(rows: list, key: str = "ret_next") -> list:
    """每个预测日等权组合收益作 1 个独立观测（消除同日横截面相关）。"""
    bucket: dict[str, list] = {}
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        bucket.setdefault(r["pred_day"], []).append(v)
    return [st.mean(v) for _, v in sorted(bucket.items()) if v]


def _eval(rows: list, label: str) -> dict:
    dayxs = _by_day_series(rows, "ret_next")
    s = _tstat(dayxs)
    win = round(100.0 * sum(1 for x in dayxs if x > 0) / len(dayxs), 1) if dayxs else 0.0
    return {"label": label, "n_trades": len(rows),
            "n_days": s["n"], "mean_pct_day": s["mean"], "sd": s["sd"],
            "t": s["t"], "win_day_pct": win, "significant": s["significant"]}


def main():
    if not RAW.exists():
        raise SystemExit(f"找不到 {RAW}；请先跑 walkforward_backtest.py --dump-raw")
    blob = json.load(open(RAW, encoding="utf-8"))
    rows = blob["rows"]
    print(f"【均值回归买点验证】raw 区间 {blob['range']['start']}~{blob['range']['end']} "
          f"({blob['range']['trading_days']}交易日) | 逐笔 {len(rows)} 条\n")

    not_down = lambda r: not r.get("gate_down")  # 非大盘下跌趋势（route-A 已压）

    # ── 基线 ──
    base_orig = [r for r in rows if r["orig_signal"] in BUY_SIGNALS]           # 原始公式买入意图
    base_range = [r for r in base_orig if r["regime"] == "range" and not_down(r)]  # 闸门只放行 range

    # ── 均值回归买点候选 ──
    rules = {
        "MR_A 动量回归": lambda r: r["momentum_20d"] < 0 and r["regime"] == "range" and not_down(r),
        "MR_B 超卖RSI<40": lambda r: r["rsi"] < 40 and r["regime"] == "range" and not_down(r),
        "MR_C 下轨<0.35": lambda r: r["bollinger_position"] < 0.35 and r["regime"] == "range" and not_down(r),
        "MR_D 深跌/超卖": lambda r: (r["momentum_20d"] < -0.03 or r["rsi"] < 35)
                                     and r["regime"] == "range" and not_down(r),
        "MR_E 大盘跌日反抽": lambda r: r["idx_ret_pct"] < -0.5 and not_down(r),
        "MR_F 任意市动量回归": lambda r: r["momentum_20d"] < 0 and not_down(r),
        "MR_G 缩量超卖": lambda r: r["rsi"] < 38 and r["volume_ratio"] < 1.2
                                     and r["regime"] == "range" and not_down(r),
    }

    results = []
    results.append(_eval(base_orig, "基线·原始公式买入(全部regime)"))
    results.append(_eval(base_range, "基线·闸门放行(range)"))
    for lbl, fn in rules.items():
        grp = [r for r in rows if fn(r)]
        results.append(_eval(grp, lbl))

    print(f"{'规则':<22}{'笔数':>6}{'日数':>5}{'均值/日':>10}{'sd':>8}{'t':>8}{'胜日%':>8}  显著")
    print("-" * 78)
    for r in results:
        flag = "★显著" if r["significant"] else ""
        print(f"{r['label']:<20}{r['n_trades']:>6}{r['n_days']:>5}"
              f"{r['mean_pct_day']:>+10.3f}{r['sd']:>8.2f}{r['t']:>+8.2f}"
              f"{r['win_day_pct']:>8.1f}  {flag}")

    best = max((r for r in results if r["label"].startswith("MR")),
               key=lambda r: r["t"], default=None)
    mr = [r for r in results if r["label"].startswith("MR")]
    sig_pos = [r for r in mr if r["significant"] and r["t"] > 0]
    sig_neg = [r for r in mr if r["significant"] and r["t"] < 0]
    print("\n=== 结论 ===")
    if sig_pos:
        for r in sig_pos:
            print(f"  ✅ {r['label']}: t={r['t']:+.2f} 显著正 edge（均值/日 {r['mean_pct_day']:+.3f}%）")
    elif sig_neg:
        print(f"  ❌ 全部显著(|t|>2)的均值回归规则 t 均为【负】——买入超卖/跌后反而显著亏钱：")
        for r in sig_neg:
            print(f"     {r['label']}: t={r['t']:+.2f} 均值/日 {r['mean_pct_day']:+.3f}% 胜日{r['win_day_pct']}%")
        print("  → ① 证伪（且更糟：均值回归买入显著负，比原公式亏更多）。"
              "价量类长仅双向都无 edge，必须转 ② 建真因子（基本面/资金流）。")
    else:
        print(f"  ❌ 无任何均值回归规则越过 |t|>2。最佳={best['label']} t={best['t']:+.2f}")
        print("  → 价量类均值回归在现有数据里也无 edge，转向 ② 建真因子（基本面/资金流）。")


if __name__ == "__main__":
    main()
