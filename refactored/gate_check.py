"""路线 A(大盘趋势闸门) 有效性直接验证 —— 股神拍板 #002 遗留项。

为什么单独验:
  拍板 #002 已证伪「个股选股 alpha」与「大盘双均线择时 beta」，
  但路线 A 闸门当前**仍在生产运行**，不能靠类比定罪，必须直接验。

问的问题:
  1. 闸门压掉的那批买入，其超额收益 alpha 是否**显著为负**？
     —— 显著为负 = 闸门确实避开了差的，有价值；否则 = 在瞎压。
  2. 应用闸门后的组合 alpha，是否优于无闸门组合？
     —— 提升 = 闸门有效；否则 = 无用功。

口径（与生产严格一致，防自欺）:
  - gate 用 `_index_gate_for_date(closes_map, anchor_day)`：决策时点已可得，无未来函数
  - 收益用 **alpha = ret_next − idx_2d**（扣掉对齐的大盘基准），不用绝对收益
  - 主判据按预测日等权组合（消除同日横截面相关）
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from akshare_fallback import _fetch_sina_kline          # noqa: E402
from backtest_signals import _index_gate_for_date       # noqa: E402

RAW = Path("data/walkforward_raw.json")
BUY_SIGNALS = ("买入", "强烈买入")


def _tstat(xs: list) -> dict:
    n = len(xs)
    if n < 2:
        return {"n": n, "mean": (xs[0] if n == 1 else 0.0), "sd": 0.0,
                "t": 0.0, "significant": False}
    m, sd = st.mean(xs), st.stdev(xs)
    se = sd / math.sqrt(n) if sd else 0.0
    t = (m / se) if se else 0.0
    return {"n": n, "mean": round(m, 4), "sd": round(sd, 4),
            "t": round(t, 2), "significant": abs(t) > 2.0}


def _by_day(rows: list, key: str) -> list:
    b: dict[str, list] = {}
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        b.setdefault(r["pred_day"], []).append(v)
    return [st.mean(v) for _, v in sorted(b.items()) if v]


def _eval(rows: list, label: str) -> dict:
    ad = _by_day(rows, "alpha")
    s = _tstat(ad)
    ab = _by_day(rows, "ret_next")
    a = _tstat(ab)
    win = (round(100.0 * sum(1 for x in ad if x > 0) / len(ad), 1) if ad else 0.0)
    return {"label": label, "n_trades": len(rows), "n_days": s["n"],
            "abs_pct": a["mean"], "alpha_pct": s["mean"], "t": s["t"],
            "win_day_pct": win, "significant": s["significant"]}


def main():
    if not RAW.exists():
        raise SystemExit(f"找不到 {RAW}")
    raw = json.load(open(RAW, encoding="utf-8"))
    rows = raw["rows"]

    # 指数历史 → gate 判定
    kl = _fetch_sina_kline("sh000001", 3000)
    closes_map = {k["day"]: float(k["close"]) for k in kl if k.get("close")}
    print(f"指数: {min(closes_map)} ~ {max(closes_map)}  {len(closes_map)} 根")

    # 对齐基准：ret_next 持有 2 日 → 拼接两日指数收益
    idx_by_day = {}
    for r in rows:
        if r.get("idx_ret_pct") is not None:
            idx_by_day[r["pred_day"]] = float(r["idx_ret_pct"])
    days = sorted(idx_by_day)
    nxt = {d: days[i + 1] for i, d in enumerate(days[:-1])}

    keep = []
    for r in rows:
        d2 = nxt.get(r["pred_day"])
        if d2 is None:
            continue
        a = float(r.get("idx_ret_pct") or 0.0) / 100.0
        b = float(idx_by_day[d2]) / 100.0
        r["idx_2d"] = round(((1 + a) * (1 + b) - 1) * 100, 4)
        r["alpha"] = round(float(r["ret_next"]) - r["idx_2d"], 4)
        r["gate"] = _index_gate_for_date(closes_map, r["anchor_day"])
        keep.append(r)

    buy = [r for r in keep if r["signal"] in BUY_SIGNALS]
    passed = [r for r in buy if r["gate"] != "down_trend"]     # 闸门放行
    blocked = [r for r in buy if r["gate"] == "down_trend"]    # 闸门压掉

    print(f"\n买入信号 {len(buy)} 笔 → 闸门放行 {len(passed)} | 压掉 {len(blocked)}\n")
    print(f"{'组别':<26}{'笔数':>7}{'日数':>6}{'绝对%/日':>10}"
          f"{'ALPHA%/日':>11}{'t':>8}{'胜日%':>7}  显著")
    print("-" * 78)
    out = []
    for label, g in [("无闸门·全部买入", buy),
                     ("闸门放行(实际会买)", passed),
                     ("闸门压掉(实际不买)", blocked)]:
        if len(g) < 2:
            print(f"{label:<24}{len(g):>7}  样本不足，跳过")
            continue
        s = _eval(g, label)
        out.append(s)
        print(f"{label:<24}{s['n_trades']:>7}{s['n_days']:>6}{s['abs_pct']:>+10.3f}"
              f"{s['alpha_pct']:>+11.3f}{s['t']:>+8.2f}{s['win_day_pct']:>7.1f}"
              f"  {'★显著' if s['significant'] else ''}")

    print("\n=== 判决 ===")
    d = {x["label"]: x for x in out}
    blk = d.get("闸门压掉(实际不买)")
    pas = d.get("闸门放行(实际会买)")
    allb = d.get("无闸门·全部买入")

    if blk:
        if blk["significant"] and blk["alpha_pct"] < 0:
            print(f"  ✅ 闸门有价值：被压掉的买入 alpha {blk['alpha_pct']:+.3f}%/日 "
                  f"t={blk['t']:+.2f} 显著为负 → 确实避开了负 alpha 的买入。")
        elif blk["significant"] and blk["alpha_pct"] > 0:
            print(f"  ❌ 闸门有害：被压掉的买入 alpha {blk['alpha_pct']:+.3f}%/日 "
                  f"t={blk['t']:+.2f} 显著为正 → 把赚钱的买盘压掉了。")
        else:
            print(f"  ❌ 闸门无证据支持：被压掉的买入 alpha {blk['alpha_pct']:+.3f}%/日 "
                  f"t={blk['t']:+.2f} 不显著 → 与 0 无差异，压掉它没有依据。")
    if pas and allb:
        diff = pas["alpha_pct"] - allb["alpha_pct"]
        print(f"  · 应用闸门后组合 alpha {pas['alpha_pct']:+.3f}%/日 vs "
              f"无闸门 {allb['alpha_pct']:+.3f}%/日，差 {diff:+.3f}%/日"
              f"（均不显著则不构成提升证据）")

    Path("data/gate_check_report.json").write_text(json.dumps(
        {"groups": out}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
