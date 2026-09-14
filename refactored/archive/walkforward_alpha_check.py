"""walk-forward 超额收益（alpha）诊断 —— 基准对齐版。

背景（why this exists）:
  walkforward_backtest.py / walkforward_meanreversion.py 的全部结论都基于**绝对收益**
  ret_next，而报告的「大盘上涨日/下跌日」分层只用了 idx_ret_pct 做**方向分组**，
  从未把指数收益从组合收益里扣掉。这导致：
    - 「大盘上涨日 +0.57%/日 t=2.77 显著盈利」这一唯一的正信号，
      无法区分「选股 alpha」与「仅仅跟着大盘涨」。
    - 若同期指数涨幅 >= 组合涨幅，则该结论不仅无 alpha，实为负 alpha。

关键口径修正（bug fix）:
  ret_next = closes[i+2] / closes[i] - 1   → 实际持有 **2 个交易日**
  idx_ret_pct = idx_ret[days[i+1]]         → 只有 **1 天** 指数涨幅
  两者口径不一致。本脚本拼接 days[i+1] 与 days[i+2] 两日指数收益（几何复合），
  构造与 ret_next 严格对齐的 2 日基准收益 idx_2d，再算 alpha = ret_next - idx_2d。

主判据沿用「按预测日等权组合」(per-day)，消除同日横截面相关。

用法:
  /Users/thinkway/.workbuddy/binaries/python/versions/3.13.12/bin/python3 \
      refactored/walkforward_alpha_check.py
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


def _by_day(rows: list, key: str) -> list:
    """每个预测日等权组合收益作 1 个独立观测。"""
    bucket: dict[str, list] = {}
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        bucket.setdefault(r["pred_day"], []).append(v)
    return [st.mean(v) for _, v in sorted(bucket.items()) if v]


def _eval(rows: list, label: str) -> dict:
    abs_day = _by_day(rows, "ret_next")
    a = _tstat(abs_day)
    alp_day = _by_day(rows, "alpha")
    b = _tstat(alp_day)
    bench = st.mean([r["idx_2d"] for r in rows]) if rows else 0.0
    win = (round(100.0 * sum(1 for x in alp_day if x > 0) / len(alp_day), 1)
           if alp_day else 0.0)
    return {"label": label, "n_trades": len(rows), "n_days": a["n"],
            "abs_mean_pct": a["mean"], "abs_t": a["t"],
            "bench_mean_pct": round(bench, 4),
            "alpha_mean_pct": b["mean"], "alpha_sd": b["sd"],
            "alpha_t": b["t"], "alpha_win_day_pct": win,
            "alpha_significant": b["significant"]}


def main():
    if not RAW.exists():
        raise SystemExit(f"找不到 {RAW}；请先跑 walkforward_backtest.py --dump-raw")

    blob = json.load(open(RAW, encoding="utf-8"))
    rows = blob["rows"]
    rng = blob["range"]
    print(f"【超额收益 alpha 诊断】raw 区间 {rng['start']}~{rng['end']} "
          f"({rng['trading_days']}交易日) | 逐笔 {len(rows)} 条")
    print("基准口径修正: ret_next 持有 2 日，idx_2d = 复合(第1日, 第2日)指数收益\n")

    # ── 建 交易日 → 当日指数收益 映射 ──
    idx_by_day: dict[str, float] = {}
    for r in rows:
        v = r.get("idx_ret_pct")
        if v is not None:
            idx_by_day[r["pred_day"]] = float(v)

    days = sorted(idx_by_day)
    next_day = {d: days[i + 1] for i, d in enumerate(days[:-1])}

    # ── 为每行补上与 ret_next 对齐的 2 日基准 ──
    usable, dropped = [], 0
    for r in rows:
        d1 = r["pred_day"]
        d2 = next_day.get(d1)
        if d2 is None:  # 最后一个预测日没有下一天，无法构造 2 日基准
            dropped += 1
            continue
        a = float(r.get("idx_ret_pct") or 0.0) / 100.0
        b = float(idx_by_day[d2]) / 100.0
        r["idx_2d"] = round(((1 + a) * (1 + b) - 1) * 100, 4)
        r["alpha"] = round(float(r["ret_next"]) - r["idx_2d"], 4)
        usable.append(r)
    if dropped:
        print(f"  ⚠️ 丢弃末日样本 {dropped} 条（无下一交易日指数，无法构造对齐基准）")
    print(f"  可用样本 {len(usable)} 条\n")

    buy = [r for r in usable if r["signal"] in BUY_SIGNALS]

    groups = [
        ("全部买入(原始公式)", [r for r in buy]),
        ("买入·无闸门压制的range日", [r for r in buy if not r.get("a_suppressed")]),
        ("买入·大盘上涨日", [r for r in buy if r["idx_ret_pct"] > 0]),
        ("买入·大盘下跌日", [r for r in buy if r["idx_ret_pct"] <= 0]),
        ("持有(对照)", [r for r in usable if r["signal"] == "持有"]),
        ("卖出(对照)", [r for r in usable if r["signal"] in ("卖出", "强烈卖出")]),
    ]

    print(f"{'组别':<26}{'笔数':>6}{'日数':>5}{'绝对%/日':>10}{'基准%/日':>10}"
          f"{'ALPHA%/日':>11}{'t':>7}{'胜日%':>7}  显著")
    print("-" * 88)
    results = []
    for label, g in groups:
        if len(g) < 2:
            continue
        s = _eval(g, label)
        results.append(s)
        print(f"{label:<24}{s['n_trades']:>7}{s['n_days']:>5}"
              f"{s['abs_mean_pct']:>+10.3f}{s['bench_mean_pct']:>+10.3f}"
              f"{s['alpha_mean_pct']:>+11.3f}{s['alpha_t']:>+7.2f}"
              f"{s['alpha_win_day_pct']:>7.1f}"
              f"  {'★' if s['alpha_significant'] else ''}")

    # 全池等权基准（不做任何选择）：衡量"随便买这20只"的 alpha
    allrows = usable
    s_all = _eval(allrows, "全池等权(不做选择)")
    print(f"{'全池等权(不做选择)':<22}{s_all['n_trades']:>7}{s_all['n_days']:>5}"
          f"{s_all['abs_mean_pct']:>+10.3f}{s_all['bench_mean_pct']:>+10.3f}"
          f"{s_all['alpha_mean_pct']:>+11.3f}{s_all['alpha_t']:>+7.2f}"
          f"{s_all['alpha_win_day_pct']:>7.1f}"
          f"  {'★' if s_all['alpha_significant'] else ''}")
    results.append(s_all)

    print("\n=== 判决 ===")
    hit = [r for r in results if r["alpha_significant"]]
    if not hit:
        print("  ❌ 没有任何分组的【超额收益】达到 |t|>2：")
        for r in results:
            print(f"     {r['label']}: alpha {r['alpha_mean_pct']:+.3f}%/日 "
                  f"t={r['alpha_t']:+.2f} 不显著")
        print("  → 此前『大盘上涨日显著盈利』经基准对齐后不成立：那是大盘 beta，不是选股 alpha。")
    else:
        for r in hit:
            direction = "正 alpha(可考虑)" if r["alpha_mean_pct"] > 0 else "负 alpha(反向可用)"
            print(f"  ⚠️ {r['label']}: alpha {r['alpha_mean_pct']:+.3f}%/日 "
                  f"t={r['alpha_t']:+.2f} {direction}")
        print("  ❗ 单个区间内的显著结果须做 OOS + 多重检验校正后再采信，不得直接上线。")

    # ── 口径 B: ret_intra（持有 1 日）与 idx_ret_pct 同日，天然对齐，无需拼接 ──
    # 生产是早盘出信号当日买入，此口径更贴近"信号日当天这一段"的收益归属。
    print("\n【口径 B · 持有 1 日（ret_intra vs 同日指数，天然对齐）】")
    print(f"{'组别':<26}{'笔数':>6}{'日数':>5}{'绝对%/日':>10}"
          f"{'基准%/日':>10}{'ALPHA%/日':>11}{'t':>7}{'胜日%':>7}  显著")
    print("-" * 82)
    intra_rows = []
    for label, g in [("全部买入(原始公式)", buy),
                     ("买入·大盘上涨日", [r for r in buy if r["idx_ret_pct"] > 0]),
                     ("买入·大盘下跌日", [r for r in buy if r["idx_ret_pct"] <= 0]),
                     ("全池等权(不做选择)", usable)]:
        if len(g) < 2:
            continue
        for r in g:  # ret_intra 与 idx_ret_pct 同为 pred_day 当日，直接相减
            r["alpha_intra"] = round(float(r.get("ret_intra") or 0.0)
                                     - float(r.get("idx_ret_pct") or 0.0), 4)
        ad = _by_day(g, "alpha_intra")
        s = _tstat(ad)
        bd = _by_day(g, "idx_ret_pct")
        bs = _tstat(bd)
        win = (round(100.0 * sum(1 for x in ad if x > 0) / len(ad), 1) if ad else 0.0)
        intra_rows.append({"label": label, "alpha_mean_pct": s["mean"],
                           "alpha_t": s["t"], "significant": s["significant"]})
        print(f"{label:<24}{len(g):>7}{s['n']:>5}{_tstat(_by_day(g,'ret_intra'))['mean']:>+10.3f}"
              f"{bs['mean']:>+10.3f}{s['mean']:>+11.3f}{s['t']:>+7.2f}{win:>7.1f}"
              f"  {'★' if s['significant'] else ''}")

    print("\n=== 口径 B 判决 ===")
    hitb = [r for r in intra_rows if r["significant"]]
    if not hitb:
        print("  ❌ 持有 1 日口径下同样无任何分组 alpha 达 |t|>2 → 结论一致：无选股 alpha。")
    else:
        for r in hitb:
            print(f"  ⚠️ {r['label']}: alpha {r['alpha_mean_pct']:+.3f}%/日 "
                  f"t={r['alpha_t']:+.2f}")

    out = Path("data/alpha_check_report.json")
    out.write_text(json.dumps({
        "generated_at": rng["start"],
        "range": rng,
        "alignment": "ret_next(2日) vs idx_2d=复合(第1日,第2日)指数收益",
        "groups": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()
