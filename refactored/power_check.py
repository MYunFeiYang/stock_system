"""统计功效分析 + 横截面多空检验 —— 回答"为什么永远测不出 alpha"。

核心问题:
  拍板 #002/#003 得出"无选股 alpha"。但该结论的**检验灵敏度**从未被检验过：
  若样本量只够检出年化 38% 的 alpha，那么"不显著"只是"检不出"，而非"不存在"。

本脚本做三件事:
  1. 算当前单边买入组合的**最小可检出效应(MDE)**：|t|=2 时需要多大的 alpha。
  2. 算横截面**多空组合**（每日按 final_score 做多 top-half / 做空 bottom-half）的
     日收益序列 —— 对冲掉市场 beta 后波动应大幅下降，检验灵敏度随之提升。
  3. 对比两种口径的 MDE，回答"要测出年化 X% 的 alpha 各需多少年数据"。

用法:
  /Users/thinkway/.workbuddy/binaries/python/versions/3.13.12/bin/python3 \
      refactored/power_check.py
"""
from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

RAW = Path("data/walkforward_raw.json")
ANNUAL_DAYS = 240


def _tstat(xs: list) -> dict:
    n = len(xs)
    if n < 2:
        return {"n": n, "mean": 0.0, "sd": 0.0, "t": 0.0, "se": 0.0,
                "significant": False}
    m, sd = st.mean(xs), st.stdev(xs)
    se = sd / math.sqrt(n) if sd else 0.0
    t = (m / se) if se else 0.0
    return {"n": n, "mean": round(m, 4), "sd": round(sd, 4),
            "t": round(t, 2), "se": round(se, 4), "significant": abs(t) > 2.0}


def mde(sd: float, n: int, t_thresh: float = 2.0) -> float:
    """|t|=t_thresh 时可检出的最小日均 alpha(%)。"""
    return t_thresh * sd / math.sqrt(n) if n > 0 else float("nan")


def years_needed(sd: float, target_daily_alpha: float,
                 t_thresh: float = 2.0) -> float:
    """检出给定日均 alpha 所需年数（每年 240 个独立日观测）。"""
    if target_daily_alpha <= 0:
        return float("nan")
    n = (t_thresh * sd / target_daily_alpha) ** 2
    return n / ANNUAL_DAYS


def main():
    if not RAW.exists():
        raise SystemExit(f"找不到 {RAW}")
    rows = json.load(open(RAW, encoding="utf-8"))["rows"]

    # 同一 pred_day 的横截面
    by_day: dict[str, list] = {}
    for r in rows:
        if r.get("ret_next") is None:
            continue
        by_day.setdefault(r["pred_day"], []).append(r)

    # ── 1. 单边买入组合（此前口径）──
    long_days = []
    for d, g in sorted(by_day.items()):
        buys = [x["ret_next"] for x in g if x["signal"] in ("买入", "强烈买入")]
        if buys:
            long_days.append(st.mean(buys))
    s_long = _tstat(long_days)

    # ── 2. 横截面多空：按 final_score 排序，做多 top-half / 做空 bottom-half ──
    ls_days, ls_n = [], []
    for d, g in sorted(by_day.items()):
        if len(g) < 6:
            continue
        g = sorted(g, key=lambda x: float(x.get("final_score") or 0))
        k = max(1, len(g) // 2)
        lo = [x["ret_next"] for x in g[:k]]
        hi = [x["ret_next"] for x in g[-k:]]
        ls_days.append(st.mean(hi) - st.mean(lo))   # 多空收益差（已对冲市场 beta）
        ls_n.append(k)
    s_ls = _tstat(ls_days)

    # ── 3. 全池等权（市场基准代理）──
    all_days = [st.mean([x["ret_next"] for x in g]) for _, g in sorted(by_day.items())]
    s_all = _tstat(all_days)

    print("=" * 76)
    print("统计功效分析：此前『无 alpha』到底是『不存在』还是『检不出』？")
    print("=" * 76)
    print(f"样本: {len(by_day)} 个交易日 | 每交易日约 {len(rows)//max(1,len(by_day))} 只")
    print(f"\n{'口径':<26}{'日数':>6}{'均值%/日':>11}{'sd%/日':>9}"
          f"{'t':>8}{'MDE%/日':>10}{'MDE年化%':>11}")
    print("-" * 82)
    for label, s in [("单边买入组合(旧口径)", s_long),
                     ("横截面多空(对冲后)", s_ls),
                     ("全池等权(市场)", s_all)]:
        m = mde(s["sd"], s["n"])
        print(f"{label:<24}{s['n']:>7}{s['mean']:>+11.3f}{s['sd']:>9.3f}"
              f"{s['t']:>+8.2f}{m:>10.3f}{m*ANNUAL_DAYS/2:>11.1f}")
    print("  (MDE = |t|=2 时可检出的最小 alpha；收益为持 2 日，年化按 /2 折算)")

    print(f"\n【关键结论】")
    m_long = mde(s_long["sd"], s_long["n"])
    m_ls = mde(s_ls["sd"], s_ls["n"])
    print(f"  旧口径只能检出 ≥ {m_long*ANNUAL_DAYS/2:.1f}%/年 的 alpha —— "
          f"现实规模的 alpha(年化5~15%) 根本检不出。")
    print(f"  多空口径可检出 ≥ {m_ls*ANNUAL_DAYS/2:.1f}%/年 —— "
          f"灵敏度提升 {m_long/m_ls:.1f} 倍。")

    print(f"\n【要多大的样本才够？】检出下列年化 alpha 所需年数")
    print(f"{'目标年化alpha':<16}{'单边买入(年)':>16}{'横截面多空(年)':>18}")
    print("-" * 52)
    for tgt in (5, 10, 15, 20):
        da = tgt / ANNUAL_DAYS * 2      # 折算为「每 2 日」的 alpha
        y1 = years_needed(s_long["sd"], da)
        y2 = years_needed(s_ls["sd"], da)
        print(f"{tgt}%{'':<13}{y1:>16.1f}{y2:>18.1f}")

    print(f"\n【多空组合本身有没有信号？】")
    print(f"  多空日均 {s_ls['mean']:+.3f}%  t={s_ls['t']:+.2f}  "
          f"{'显著' if s_ls['significant'] else '不显著'}"
          f"  胜日 {100*sum(1 for x in ls_days if x>0)/len(ls_days):.1f}%")
    if s_ls["significant"]:
        print("  → 若显著且为正：选股能力确实存在，只是被市场 beta 淹没，"
              "架构应改为市场中性。")
    else:
        print(f"  → 不显著。但注意：多空口径 MDE 为 {m_ls*ANNUAL_DAYS/2:.1f}%/年，"
              f"结论比旧口径可靠得多。")

    Path("data/power_check_report.json").write_text(json.dumps({
        "long_only": s_long, "long_short": s_ls, "all_equal": s_all,
        "mde_long_pct_day": round(m_long, 4),
        "mde_ls_pct_day": round(m_ls, 4),
        "sensitivity_gain": round(m_long / m_ls, 2) if m_ls else None,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
