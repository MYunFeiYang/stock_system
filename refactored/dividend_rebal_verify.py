#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""高股息受控重验（严格执行 高股息受控重验_预注册.md，预注册时间戳 2026-09-07 16:41 先于本脚本任何数据访问）。

口径（预注册锁定，不可改）：
- 定位：红利当"债的补充"（配置层风险预算），非选股 alpha。
  处理组 = 60% 沪深300 + X% 上证红利 + (40−X)% 国债（X∈{10,20} 参数平面全报），
  vs 基准 = 60/40。年度再平衡+漂移，双边 0.2% 成本（净敞口×0.1%×2）。
- 主决策 = 增量收益对风格因子正交化后的**残差 t_NW**（改进③）：
  增量 d[t]=处理−基准，滚动 244 日窗口 OLS 剥离 市场(300)/规模(852)/成长(399006)/大盘价值(000016)，
  残差均值 t_NW>2 且正才算数。零未来函数（系数只用 t 日前数据）。
- 已知局限：低波/盈利代理缺失（免费缓存无），若残差显著需后续补验。
- 股息口径 STOCK_DIV=2 统一补（红利实际股息率~5%>2%，偏差不利于红利=更严格）。
"""
import json
import statistics as st
from pathlib import Path

import backtest_lib as bl

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "dividend_rebal_report.json"
ORTH_WIN = 244        # 正交化滚动窗口
CUT_DATE = "2015-01-01"


def solve_ls(X, y):
    """最小二乘正规方程 (X'X)b=X'y，X 含截距列。高斯消元。"""
    k = len(X[0])
    A = [[sum(X[r][i] * X[r][j] for r in range(len(X))) for j in range(k)]
         + [sum(X[r][i] * y[r] for r in range(len(X)))] for i in range(k)]
    for c in range(k):
        p = max(range(c, k), key=lambda r: abs(A[r][c]))
        A[c], A[p] = A[p], A[c]
        pv = A[c][c]
        if abs(pv) < 1e-12:
            return None
        A[c] = [v / pv for v in A[c]]
        for r in range(k):
            if r != c and A[r][c]:
                f = A[r][c]
                A[r] = [v - f * w for v, w in zip(A[r], A[c])]
    return [A[i][k] for i in range(k)]


def main():
    syms = ["sh000300", "sh000015", "sh000012", "sh000905",
            "sz399006", "sh000016"]
    data = {s: bl.load(s) for s in syms}
    dates = sorted(set.intersection(*[set(d) for d in data.values()]))
    n = len(dates)
    rets = {s: bl.daily_returns(data[s], dates,
                                bl.DIV_D if s != "sh000012" else 0.0)
            for s in syms}
    factors = [rets["sh000300"], rets["sh000905"],
               rets["sz399006"], rets["sh000016"]]
    print(f"样本 {dates[0]}~{dates[-1]} ({n/244:.1f}年, {n}日) "
          f"股息+{bl.DIV*100:.0f}%")

    base = bl.portfolio_daily(rets, {"sh000300": 0.60, "sh000012": 0.40})
    results = {"meta": {"start": dates[0], "end": dates[-1], "days": n,
                        "cost": bl.COST, "div": bl.DIV, "orth_win": ORTH_WIN},
               "rows": []}
    cut = next(i for i, d in enumerate(dates) if d >= CUT_DATE)
    print(f"\n{'X红利%':<8}{'口径':<10}{'CAGR差':>9}{'MDD处/基':>16}"
          f"{'超额均值':>11}{'t_ord':>8}{'t_NW':>8}{'pre_NW':>9}{'post_NW':>9}")
    for X in (0.10, 0.20):
        treat = bl.portfolio_daily(rets, {"sh000300": 0.60, "sh000015": X,
                                          "sh000012": 0.40 - X})
        d = [treat[i] - base[i] for i in range(1, n)]      # 含成本增量
        for label in ("原始", "正交"):
            if label == "正交":
                # 滚动窗口正交化：残差 = d − (a + b·f)，系数只用 t 日前 ORTH_WIN 日
                resid = [None] * len(d)
                for t in range(ORTH_WIN, len(d)):
                    Xmat = [[1.0] + [f[i + 1] for f in factors]
                            for i in range(t - ORTH_WIN, t)]
                    yv = [d[i] for i in range(t - ORTH_WIN, t)]
                    coef = solve_ls(Xmat, yv)
                    if coef is None:
                        resid[t] = 0.0
                        continue
                    pred = sum(c * v for c, v in zip(
                        coef, [1.0] + [f[t + 1] for f in factors]))
                    resid[t] = d[t] - pred
                seq = [x * 100 for x in resid if x is not None]
                dseg = resid
            else:
                seq = [x * 100 for x in d]
                dseg = d
            t_o, t_nw, q = bl.tstat_nw(seq)
            pre = [x for i, x in enumerate(dseg) if i + 1 < cut and x is not None]
            post = [x for i, x in enumerate(dseg) if i + 1 >= cut and x is not None]
            _, pre_t, _ = bl.tstat_nw([x * 100 for x in pre])
            _, post_t, _ = bl.tstat_nw([x * 100 for x in post])
            pt = bl.perf(treat, (n - 1) / 244)
            pb = bl.perf(base, (n - 1) / 244)
            tc, tm, bc, bm = pt["cagr"], pt["mdd"], pb["cagr"], pb["mdd"]
            print(f"{X*100:<8.0f}{label:<10}{tc-bc:>+8.2f}%{tm:>7.1f}/{bm:<7.1f}"
                  f"{st.mean(seq):>+10.4f}%{t_o:>+8.2f}{t_nw:>+8.2f}"
                  f"{pre_t:>+9.2f}{post_t:>+9.2f}")
            results["rows"].append({
                "x": X, "mode": label, "cagr_diff_pct": round(tc - bc, 2),
                "mdd_treat": round(tm, 1), "mdd_base": round(bm, 1),
                "excess_mean_pct": round(st.mean(seq), 4),
                "t_ord": round(t_o, 2), "t_nw": round(t_nw, 2),
                "t_pre_nw": round(pre_t, 2), "t_post_nw": round(post_t, 2)})
    # 门槛判定（主口径=正交化残差）
    orth_rows = [r for r in results["rows"] if r["mode"] == "正交"]
    g1 = all(r["t_nw"] > 2 and r["excess_mean_pct"] > 0 for r in orth_rows)
    g2 = all(r["excess_mean_pct"] > 0 for r in orth_rows)   # 已含0.2%成本
    g3 = all(abs(r["t_pre_nw"]) > 1.5 and abs(r["t_post_nw"]) > 1.5
             and (r["t_pre_nw"] > 0) == (r["t_post_nw"] > 0) for r in orth_rows)
    verdict = ("PASS 全门槛→可纳入配置层(债的补充)" if (g1 and g2 and g3)
               else f"FAIL → 维持冻结 (①{g1} ②{g2} ③{g3})")
    results["gates"] = {"alpha": g1, "cost_net": g2, "subperiod": g3}
    results["verdict"] = verdict
    results["known_limitation"] = ("低波/盈利因子代理缺失; 规模用中证500代理"
                                   "(中证1000自2014-10发布, 会砍短样本故剔除)")
    print(f"\n门槛① 逐日超额残差|t_NW|>2且正 : {'PASS' if g1 else 'FAIL'}")
    print(f"门槛② 扣双边0.2%后净正       : {'PASS' if g2 else 'FAIL'}")
    print(f"门槛③ 子样本同号且|t_NW|>1.5 : {'PASS' if g3 else 'FAIL'}")
    print(f"→ {verdict}")
    print(f"局限: {results['known_limitation']}")
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"已写 {OUT.name}")


if __name__ == "__main__":
    main()
