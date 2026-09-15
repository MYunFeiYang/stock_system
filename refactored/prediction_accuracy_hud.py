#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预测准确率看板 (HUD)。

直接从累计对账数据计算真实口径，不依赖报告文本正则（避免抓空显示 "—%"）。

数据来源（均已在系统中每日累计）：
  - data/reconcile_history.jsonl      每日下午对账（predict→actual）
  - data/rs_prediction_scorecard.json 相对强弱动量信号记分卡
  - reports/day_review_report_*.txt   收盘复盘报告（含「股神淘汰线」回测判据）

设计原则：只报真实口径，不报被「持有」信号注水的虚高一致率。
口径说明：
  · 累计方向一致率 = Σ命中 / Σ(命中+未命中)，样本量加权，含「持有」信号
  · 剔除持有注水   = 只统计有方向性（买入/卖出）的信号，样本量加权
  · 近5次          = 最近 5 个交易日一致率的算术平均
  · z 值           = 累计命中率相对 50%（硬币线）的正态近似，|z|>2 为显著
"""
import json
import math
import glob
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"


def latest_report():
    fs = sorted(glob.glob(str(REPORTS / "day_review_report_*.txt")))
    return Path(fs[-1]) if fs else None


def load_reconcile_rows():
    p = DATA / "reconcile_history.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def compute_accuracy():
    """返回真实口径的准确率统计 dict。"""
    rows = load_reconcile_rows()
    stats = {
        "n_days": len(rows),
        "trade_date": rows[-1].get("trade_date") if rows else None,
        "cum_hit": None, "cum_n": 0, "cum_z": None,
        "ex_hold_hit": None, "ex_hold_n": 0,
        "recent5": None,
    }
    if not rows:
        return stats

    hits = sum(r.get("stats", {}).get("direction_hits", 0) or 0 for r in rows)
    miss = sum(r.get("stats", {}).get("direction_misses", 0) or 0 for r in rows)
    n = hits + miss
    if n:
        p = hits / n
        stats["cum_hit"] = p
        stats["cum_n"] = n
        stats["cum_z"] = (p - 0.5) / math.sqrt(0.25 / n)

    ds = 0
    dh = 0.0
    for r in rows:
        s = r.get("stats", {})
        m = s.get("directional_samples") or 0
        hr = s.get("directional_hit_rate")
        if m and hr is not None:
            ds += m
            dh += hr * m
    if ds:
        stats["ex_hold_hit"] = dh / ds
        stats["ex_hold_n"] = ds

    hr_all = [r.get("stats", {}).get("hit_rate") for r in rows]
    hr_all = [x for x in hr_all if x is not None]
    if hr_all:
        stats["recent5"] = st.mean(hr_all[-5:])
    return stats


def build_hud_lines(rep_txt=None):
    """组装看板文本行（供收盘推送与人工查看共用）。"""
    acc = compute_accuracy()
    if rep_txt is None:
        rep = latest_report()
        rep_txt = rep.read_text(encoding="utf-8") if rep else ""

    date_s = ""
    if acc["trade_date"]:
        d = str(acc["trade_date"])
        date_s = d

    def pct(x):
        return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "—"

    cum = pct(acc["cum_hit"])
    if acc["cum_z"] is not None:
        cum += f"(n={acc['cum_n']},z={acc['cum_z']:+.2f})"
    ex = pct(acc["ex_hold_hit"])
    r5 = pct(acc["recent5"])

    # 股神淘汰线：隔日买入组合 P&L（全样本）—— 报告里唯一一段，取第一个匹配
    pnl_line = "隔日买入组合 P&L: 数据缺失"
    m = None
    if rep_txt:
        import re
        m = re.search(
            r"样本\s*n=(\d+)\s*\|\s*均值=([-\d.]+)%\s*\|\s*t=([-\d.]+)\s*\|\s*显著=(\S+)",
            rep_txt,
        )
    if m:
        n_, mean_, t_, sig_ = m.groups()
        pnl_line = f"隔日买入组合 P&L: n={n_} 均值={mean_}% t={t_} 显著={sig_}"
    else:
        mean_, sig_ = "—", "—"

    # RS 动量信号记分卡
    sc = {}
    try:
        sc = json.loads((DATA / "rs_prediction_scorecard.json").read_text())
    except Exception:
        pass
    at = sc.get("all_time", {}) or {}
    rs_line = (
        f"相对强弱(RS)信号: 命中 {at.get('hit_rate', '—')} "
        f"t={at.get('t', '—')} → {sc.get('advice_level', '—')}"
    )

    # 判级（诚实口径）
    if str(sig_) == "是" and str(mean_).startswith("-"):
        verdict = "❌ 已证伪（显著亏钱，停用该信号）"
    elif acc["cum_z"] is not None and acc["cum_z"] <= -2:
        verdict = "❌ 淘汰（累计命中率显著差于硬币）"
    elif acc["cum_hit"] is not None and acc["cum_hit"] >= 0.55 and acc["cum_z"] and acc["cum_z"] >= 2:
        verdict = "✅ 过线（可关注跟单）"
    else:
        verdict = "OBSERVE（未过股神淘汰线，暂不动）"

    if acc["cum_z"] is not None and acc["cum_z"] <= -2:
        tail = f"→ 累计命中率 {pct(acc['cum_hit'])} 显著低于硬币(z={acc['cum_z']:+.2f})，按约定不动"
    else:
        tail = "→ 当前信号不过线，按约定暂不动；继续每日积累样本"

    return [
        f"📈 预测准确率看板 (截至 {date_s or '—'})",
        f"· 方向一致率 累计 {cum} ｜ 剔除持有 {ex} ｜ 近5次 {r5}",
        f"· {pnl_line}",
        f"· {rs_line}",
        f"· 判级: {verdict}",
        tail,
    ]


def main():
    print("\n".join(build_hud_lines()))


if __name__ == "__main__":
    main()
