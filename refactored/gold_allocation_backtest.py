#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""股/债/金扩池验证（严格执行 股债金扩池验证_预注册.md，时间戳先于本脚本任何收益计算）。

基准 = 60/40（沪深300+国债）；处理 = 60/(40−X)/X，X∈{10,15,20}。
年度再平衡+漂移，双边0.2%，超额=算术日收益差，t 带 Newey-West。
黄金 sh518880 价格≈全收益（ETF跟踪金价，无股息），不补股息。
组合/统计口径统一走 backtest_lib（2026-09-07 重构）。
"""
import json
import statistics as st
from pathlib import Path

import backtest_lib as bl

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "gold_alloc_report.json"
CUT_DATE = "2015-01-01"
XS = (0.10, 0.15, 0.20)


def main():
    stock, bond, gold = bl.load("sh000300"), bl.load("sh000012"), bl.load("sh518880")
    start = max(min(stock), min(bond), min(gold))
    dates = sorted(d for d in stock if d in bond and d in gold and d >= start)
    n = len(dates)
    rs = {"stock": bl.daily_returns(stock, dates, bl.DIV_D),
          "bond": bl.daily_returns(bond, dates),
          "gold": bl.daily_returns(gold, dates)}
    yrs = (n - 1) / 244
    print(f"样本 {dates[0]}~{dates[-1]} ({yrs:.1f}年, {n}日) "
          f"股息+{bl.DIV*100:.0f}% 黄金不补息")

    base = bl.portfolio_daily(rs, {"stock": 0.60, "bond": 0.40})
    pb = bl.perf(base, yrs)
    # 黄金单独画像（供机制判断）
    pg = bl.perf(bl.portfolio_daily(rs, {"gold": 1.0}), yrs)
    corr = st.correlation(rs["gold"][1:], rs["stock"][1:])
    print(f"黄金单独: 年化{pg['cagr']:+.2f}% 回撤{pg['mdd']:.1f}% "
          f"| 金-股日收益相关 {corr:+.2f}")
    print(f"基准60/40: 年化{pb['cagr']:+.2f}% 回撤{pb['mdd']:.1f}%")
    cut = next(i for i, d in enumerate(dates) if d >= CUT_DATE)
    rows = []
    print(f"\n{'X金%':<7}{'年化':>8}{'回撤':>9}{'夏普':>7}{'超额均值':>11}"
          f"{'t_NW':>8}{'pre_t':>8}{'post_t':>8}")
    for X in XS:
        tr = bl.portfolio_daily(rs, {"stock": 0.60, "bond": 0.40 - X, "gold": X})
        pt = bl.perf(tr, yrs)
        d = [tr[i] - base[i] for i in range(1, n)]
        _, t_nw, _ = bl.tstat_nw([x * 100 for x in d])
        _, pre_t, _ = bl.tstat_nw([x * 100 for x in d[:cut - 1]])
        _, post_t, _ = bl.tstat_nw([x * 100 for x in d[cut - 1:]])
        print(f"{X*100:<7.0f}{pt['cagr']:>+7.2f}%{pt['mdd']:>8.1f}%{pt['sharpe']:>7.2f}"
              f"{st.mean([x*100 for x in d]):>+10.4f}%{t_nw:>+8.2f}"
              f"{pre_t:>+8.2f}{post_t:>+8.2f}")
        rows.append({"x": X, "cagr": pt["cagr"], "mdd": pt["mdd"],
                     "sharpe": pt["sharpe"], "t_nw": round(t_nw, 2),
                     "t_pre": round(pre_t, 2), "t_post": round(post_t, 2),
                     "mdd_improve_pp": round(pb["mdd"] - pt["mdd"], 1)})
    # 门槛（预注册双轨①：收益端 t_NW>2 或 风控端 MDD改善≥5pp 且超额不显著为负）
    g1 = any((r["t_nw"] > 2) or (r["mdd_improve_pp"] >= 5 and r["t_nw"] > -2)
             for r in rows)
    g2 = all(r["t_nw"] > -2 for r in rows)
    g3 = all((r["t_pre"] > 0) == (r["t_post"] > 0) for r in rows)
    g4 = len({(r["mdd_improve_pp"] > 0) for r in rows}) == 1
    verdict = "PASS → 建议股神拍板指令化" if (g1 and g2 and g3 and g4) \
        else f"未全过 (①{g1} ②{g2} ③{g3} ④{g4}) → 归档"
    print(f"\n门槛① 收益t>2 或 MDD改善≥5pp且不显著为负 : {'PASS' if g1 else 'FAIL'}")
    print(f"门槛② 成本后无显著恶化                  : {'PASS' if g2 else 'FAIL'}")
    print(f"门槛③ 子样本方向一致(2015前仅~2年,弱证据): {'PASS' if g3 else 'FAIL'}")
    print(f"门槛④ X三档方向一致                     : {'PASS' if g4 else 'FAIL'}")
    print(f"→ {verdict}")
    OUT.write_text(json.dumps({
        "meta": {"start": dates[0], "end": dates[-1], "days": n,
                 "gold_first": dates[0], "corr_gold_stock": round(corr, 3)},
        "base": {"cagr": pb["cagr"], "mdd": pb["mdd"]},
        "gold_alone": {"cagr": pg["cagr"], "mdd": pg["mdd"]},
        "rows": rows, "gates": {"risk_or_alpha": g1, "cost": g2,
                                "subperiod": g3, "multipletest": g4},
        "verdict": verdict}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"已写 {OUT.name}")


if __name__ == "__main__":
    main()
