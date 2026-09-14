"""大盘 beta 择时 —— walk-forward 验证（股神拍板 #002 唯一批准的新方向）。

为什么做这个:
  个股选股 alpha 已被双口径证伪(见 walkforward_alpha_check.py)，收益的主宰是
  **大盘方向**而非选股。因此唯一值得验证的是「对大盘本身做择时」。

防自欺设计（写死，不许绕）:
  1. 参数**不许挑**: MA20/60 只是经济学先验(趋势跟随)，结论必须由**整个参数平面**
     的分布支撑——若只有个别参数点好看，判为过拟合。
  2. 样本外分割: IS(前 60%) / OOS(后 40%)，OOS 与 IS 必须同向才采信。
  3. 空仓收益按 **0** 计（不白送货基利息，保守）。
  4. 成本按月频/日频实际调仓次数扣，ETF 双边按 0.10%/次（保守，含滑点）。
  5. 无未来函数: signal[t] 只用 ≤t 的收盘价，收益从 t+1 起算。

用法:
  /Users/thinkway/.workbuddy/binaries/python/versions/3.13.12/bin/python3 \
      refactored/index_timing_backtest.py
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from akshare_fallback import _fetch_sina_kline  # noqa: E402

DAYS = 3000               # 约 12 年，跨 2015 股灾 / 2018 熊市
COST_PCT = 0.10           # 每次调仓双边成本(%)，ETF 保守估计（含滑点）
RF_ANNUAL = 0.02          # 无风险利率，用于 Sharpe
SYMBOLS = [("sh000001", "上证指数"), ("sh000300", "沪深300"), ("sz399006", "创业板指")]
GRID = [(10, 30), (10, 60), (20, 60), (20, 90), (30, 60),
        (30, 90), (30, 120), (20, 120), (50, 120), (10, 120)]
IS_FRAC = 0.6


def _rolling_mean(xs: list, w: int) -> list:
    out, s = [], 0.0
    for i, v in enumerate(xs):
        s += v
        if i >= w:
            s -= xs[i - w]
        out.append(s / w if i >= w - 1 else None)
    return out


def _max_drawdown(equity: list) -> float:
    peak, mdd = equity[0], 0.0
    for v in equity:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    return mdd * 100


def _annualize(total_mult: float, n_days: int) -> float:
    if n_days <= 0 or total_mult <= 0:
        return 0.0
    return (total_mult ** (252.0 / n_days) - 1) * 100


def backtest(closes: list, days: list, short: int, long: int,
             monthly: bool = False, cost_pct: float = COST_PCT,
             warmup: int | None = None, idle_annual: float = 0.0):
    """返回 (stats, rets, bh_rets, trades)。rets 为策略日收益序列(%)。

    warmup: 统一回测起点。必须对所有参数取同一个值(网格内最大 long)，
    否则不同参数的基准区间长度不同 → 基准年化漂移 → 超额不可比。
    """
    n = len(closes)
    ma_s = _rolling_mean(closes, short)
    ma_l = _rolling_mean(closes, long)

    rebal = set()
    if monthly:
        prev = None
        for i, d in enumerate(days):
            m = d[:7]
            if m != prev:
                rebal.add(i)
                prev = m

    pos, equity, trades = 0, 1.0, 0
    rets, bh_rets, eq_curve = [], [], [1.0]
    start = warmup if warmup is not None else long  # 前 long 根无 MA
    idle_daily = idle_annual / 252.0 / 100.0   # 空仓期货基收益(按日计)

    for i in range(start, n - 1):
        sig = 1 if (ma_s[i] is not None and ma_l[i] is not None
                    and ma_s[i] > ma_l[i]) else 0
        if monthly and i not in rebal:
            sig = pos
        if sig != pos:
            trades += 1
            equity *= (1 - cost_pct / 100.0)
            pos = sig
        r = closes[i + 1] / closes[i] - 1
        bh_rets.append(r * 100)
        if pos == 1:
            equity *= (1 + r)
            rets.append(r * 100)
        else:
            equity *= (1 + idle_daily)
            rets.append(idle_daily * 100)   # 空仓期货基收益
        eq_curve.append(equity)

    nbh = 1.0
    for r in bh_rets:
        nbh *= (1 + r / 100)
    nd = len(rets)
    mdd = _max_drawdown(eq_curve)
    bh_curve, c = [1.0], 1.0
    for r in bh_rets:
        c *= (1 + r / 100)
        bh_curve.append(c)

    ann = _annualize(equity, nd)
    bh_ann = _annualize(nbh, nd)
    sd = st.pstdev(rets) if nd > 1 else 0.0
    sharpe = ((st.mean(rets) * 252 / 100) - RF_ANNUAL) / (sd * math.sqrt(252) / 100) \
        if sd > 0 else 0.0

    return {
        "annual_pct": round(ann, 2),
        "bh_annual_pct": round(bh_ann, 2),
        "excess_ann_pct": round(ann - bh_ann, 2),
        "max_dd_pct": round(mdd, 1),
        "bh_max_dd_pct": round(_max_drawdown(bh_curve), 1),
        "sharpe": round(sharpe, 2),
        "trades": trades,
        "final_mult": round(equity, 3),
        "bh_final_mult": round(nbh, 3),
    }, rets, bh_rets, trades


def _paired_t(rets: list, bh: list) -> tuple:
    """择时 vs 买入持有的日收益差 → 配对 t 检验（同日配对，消除共同市场因子）。"""
    d = [a - b for a, b in zip(rets, bh)]
    n = len(d)
    if n < 2:
        return 0.0, 0.0, False
    m, sd = st.mean(d), st.stdev(d)
    se = sd / math.sqrt(n)
    t = (m / se) if se else 0.0
    return round(m, 4), round(t, 2), abs(t) > 2.0


def run_symbol(symbol: str, name: str, idle_annual: float = 0.0,
               cost: float = COST_PCT) -> dict:
    kl = _fetch_sina_kline(symbol, DAYS)
    if not kl:
        return {"symbol": symbol, "name": name, "error": "取数失败"}
    days = [k["day"] for k in kl]
    closes = [float(k["close"]) for k in kl]
    n = len(closes)

    print(f"\n{'='*78}\n{name} ({symbol})  {days[0]} ~ {days[-1]}  {n} 根"
          f"   收盘 {closes[0]:.0f} → {closes[-1]:.0f}\n{'='*78}")

    is_end = int(n * IS_FRAC)
    warm = max(l for _, l in GRID)   # 统一回测起点，保证超额可比
    results = []
    for monthly in (False, True):
        tag = "月频调仓" if monthly else "日频信号"
        print(f"\n【{tag}】")
        print(f"{'MA(s/l)':<10}{'年化%':>8}{'基准年化%':>10}{'超额%':>9}"
              f"{'最大回撤%':>10}{'基准回撤%':>11}{'Sharpe':>8}{'调仓':>6}{'日差t':>8}")
        print("-" * 82)
        for s_, l_ in GRID:
            stat, rets, bh, tr = backtest(closes, days, s_, l_, monthly=monthly,
                                          warmup=warm, idle_annual=idle_annual,
                                          cost_pct=cost)
            _, t, sig = _paired_t(rets, bh)
            stat.update({"short": s_, "long": l_, "monthly": monthly, "t": t,
                         "sig": sig})
            results.append(stat)
            print(f"{f'{s_}/{l_}':<10}{stat['annual_pct']:>8.2f}"
                  f"{stat['bh_annual_pct']:>10.2f}{stat['excess_ann_pct']:>+9.2f}"
                  f"{stat['max_dd_pct']:>10.1f}{stat['bh_max_dd_pct']:>11.1f}"
                  f"{stat['sharpe']:>8.2f}{tr:>6}{t:>8.2f}{'  *' if sig else ''}")

    # ── 样本外：用 IS 段挑参数，在 OOS 段检验 ──
    print(f"\n【样本外检验】IS = {days[0]}~{days[is_end-1]} | "
          f"OOS = {days[is_end]}~{days[-1]}")
    oos_rows = []
    for monthly in (False, True):
        tag = "月频调仓" if monthly else "日频信号"
        best = None
        for s_, l_ in GRID:
            st_is, _, _, _ = backtest(closes[:is_end], days[:is_end], s_, l_,
                                      monthly=monthly, warmup=warm,
                                      idle_annual=idle_annual)
            if best is None or st_is["annual_pct"] > best[0]["annual_pct"]:
                best = (st_is, s_, l_)
        st_is, s_, l_ = best
        # OOS: 用全段前 is_end 根做 warmup，收益只统计 OOS 段
        st_oos, rets_oos, bh_oos, _ = backtest(closes, days, s_, l_,
                                               monthly=monthly, warmup=warm,
                                               idle_annual=idle_annual)
        # 仅取 OOS 部分日收益
        cut = len(rets_oos) - (n - is_end)
        r_o, b_o = rets_oos[cut:], bh_oos[cut:]
        md, t_o, sig_o = _paired_t(r_o, b_o)
        ann_o = (st.mean(r_o) * 252) if r_o else 0.0
        ann_b = (st.mean(b_o) * 252) if b_o else 0.0
        oos_rows.append({"mode": tag, "short": s_, "long": l_,
                         "is_annual": st_is["annual_pct"],
                         "oos_annual": round(ann_o, 2),
                         "oos_bh_annual": round(ann_b, 2),
                         "oos_excess": round(ann_o - ann_b, 2),
                         "oos_t": t_o, "oos_sig": sig_o})
        print(f"  {tag}: IS 最优 MA={s_}/{l_} (IS年化 {st_is['annual_pct']:+.2f}%) "
              f"→ OOS年化 {ann_o:+.2f}% vs 基准 {ann_b:+.2f}% "
              f"超额 {ann_o-ann_b:+.2f}% t={t_o:+.2f} {'显著' if sig_o else '不显著'}")

    return {"symbol": symbol, "name": name, "days": n,
            "range": [days[0], days[-1]], "grid": results, "oos": oos_rows}


def main():
    import os
    idle = float(os.environ.get('IDLE_ANNUAL', '0'))
    cost = float(os.environ.get('COST_PCT', str(COST_PCT)))
    if cost != COST_PCT:
        print(f"\n[成本假设: {cost}%/次]")
    out = []
    for sym, name in SYMBOLS:
        out.append(run_symbol(sym, name, idle_annual=idle, cost=cost))

    print(f"\n\n{'='*78}\n【股神判据汇总】\n{'='*78}")
    print(f"{'标的':<10}{'模式':<10}{'优于基准的参数占比':>18}"
          f"{'中位超额%':>12}{'回撤改善(中位)':>16}")
    print("-" * 70)
    verdict = []
    for r in out:
        if "error" in r:
            continue
        for monthly in (False, True):
            g = [x for x in r["grid"] if x["monthly"] == monthly]
            if not g:
                continue
            win = sum(1 for x in g if x["excess_ann_pct"] > 0)
            med_ex = st.median([x["excess_ann_pct"] for x in g])
            med_dd = st.median([x["bh_max_dd_pct"] - x["max_dd_pct"] for x in g])
            tag = "月频" if monthly else "日频"
            print(f"{r['name']:<8}{tag:<10}{win}/{len(g):>15}"
                  f"{med_ex:>+12.2f}{med_dd:>+16.1f}")
            verdict.append({"symbol": r["symbol"], "name": r["name"],
                            "mode": tag, "win_rate": f"{win}/{len(g)}",
                            "median_excess": round(med_ex, 2),
                            "median_dd_improve": round(med_dd, 1)})

    p = Path("data/index_timing_report.json")
    p.write_text(json.dumps({"cost_pct_per_trade": COST_PCT,
                             "grid_tested": GRID,
                             "symbols": out, "verdict": verdict},
                            ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {p}")


if __name__ == "__main__":
    main()
