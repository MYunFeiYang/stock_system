#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配置层面的买卖框架验证：等权 4 宽基 + 再平衡规则。

动机：用户诉求是"指导买入卖出"。此前所有努力都在"预测涨跌"（全部证伪），
但**买卖指导 ≠ 预测**：配置买什么、何时调仓，靠的是再平衡（风险控制 +
被动低买高卖），有经济学先验，不依赖任何预测能力。

标的（均有 ETF，零花费，官方指数零幸存者偏差）:
  沪深300 / 上证50 / 中证500 / 上证红利
策略:
  A 买入持有（等权初始，不再平衡）
  B 年度再平衡
  C 季度再平衡
  D 阈值再平衡（权重偏离目标 ±X% 触发）
对比年化/回撤/夏普(粗算)/调仓次数/成本。
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sector_rotation_backtest import _tstat  # noqa: E402
from style_rotation import fetch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "rebalance_report.json"
SYM = {"sh000300": "沪深300", "sh000016": "上证50",
       "sh000905": "中证500", "sh000015": "上证红利"}
ETF = {"沪深300": "510300", "上证50": "510050",
       "中证500": "510500", "上证红利": "510880"}
COST = 0.0005      # ETF 单边成本(佣金, 免印花税) ≈0.05%


def main():
    data = {n: dict(fetch(s)) for s, n in SYM.items()}
    start = max(min(v) for v in data.values())
    dates = sorted({d for v in data.values() for d in v if d >= start})
    idx = {d: i for i, d in enumerate(dates)}
    arrs = {n: {idx[d]: c for d, c in v.items() if d in idx}
            for n, v in data.items()}
    names = sorted(arrs)
    n_asset = len(names)
    target = 1.0 / n_asset
    print(f"标的 {[(n, ETF[n]) for n in names]}")
    print(f"覆盖 {dates[0]} ~ {dates[-1]} ({len(dates)/244:.1f} 年)", flush=True)

    ret = {n: {} for n in names}
    for t in range(1, len(dates)):
        for n in names:
            a = arrs[n]
            if t in a and (t - 1) in a and a[t - 1] > 0:
                ret[n][t] = a[t] / a[t - 1] - 1

    def simulate(mode, param=None):
        """mode: hold / period / threshold"""
        w = {n: target for n in names}
        eq = 1.0
        eqs = []
        trades = 0
        rets = []
        for t in range(1, len(dates)):
            for n in names:
                w[n] *= (1 + ret[n].get(t, 0.0))
            tot = sum(w.values())
            w = {n: v / tot for n, v in w.items()}
            dret = tot - 1.0
            need = False
            if mode == "period" and t % param == 0:
                need = True
            elif mode == "threshold":
                for n in names:
                    dev = w[n] - target
                    if abs(dev) >= param * target:
                        need = True
                        break
            if need:
                turnover = sum(abs(w[n] - target) for n in names) / 2
                cost = turnover * COST * 2     # 双边
                eq *= (1 - cost)
                trades += 1
                w = {n: target for n in names}
            eq *= (1 + dret)
            eqs.append(eq)
            rets.append(dret * 100)
        peak, mdd = 1.0, 0.0
        for e in eqs:
            peak = max(peak, e)
            mdd = min(mdd, e / peak - 1)
        s = _tstat(rets)
        ann = s["mean"] * 252 / 100
        years = len(dates) / 244
        cagr = (eq ** (1 / years) - 1) * 100
        vol = st.pstdev(rets) * (252 ** 0.5) if rets else 0
        sharpe = (s["mean"] * 252 / 100) / vol if vol else 0
        return {"cagr": round(cagr, 2), "mdd_pct": round(mdd * 100, 1),
                "ann_vol_pct": round(vol, 1), "sharpe": round(sharpe, 2),
                "rebalances": trades}

    rows = []
    print(f"\n{'策略':<20}{'年化CAGR':>9}{'回撤':>8}{'波动':>8}{'夏普':>7}{'调仓次数':>9}",
          flush=True)
    for label, mode, param in [
        ("A 买入持有(不再平衡)", "hold", None),
        ("B 年度再平衡", "period", 244),
        ("C 季度再平衡", "period", 61),
        ("D 阈值±25%再平衡", "threshold", 0.25),
        ("E 阈值±50%再平衡", "threshold", 0.50),
    ]:
        r = simulate(mode, param)
        rows.append({"strategy": label, **r})
        print(f"{label:<20}{r['cagr']:>+8.2f}%{r['mdd_pct']:>7.1f}%"
              f"{r['ann_vol_pct']:>7.1f}%{r['sharpe']:>7.2f}{r['rebalances']:>9}",
              flush=True)

    OUT.write_text(json.dumps({"start": dates[0], "end": dates[-1],
                               "assets": [(n, ETF[n]) for n in names],
                               "rows": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    best = max(rows, key=lambda r: r["sharpe"])
    print(f"\n风险调整后最优: {best['strategy']} "
          f"(夏普{best['sharpe']}, CAGR{best['cagr']}%, 回撤{best['mdd_pct']}%, "
          f"{best['rebalances']}次调仓)")
    print(f"已写 {OUT.name}", flush=True)


if __name__ == "__main__":
    main()
