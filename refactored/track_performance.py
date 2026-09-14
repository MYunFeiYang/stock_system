#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实盘跟踪：按系统指令操作的组合每日盯市 + 基准对比。

- 组合 = data/portfolio.json 的 holdings/cash 每日盯市（调仓自动反映，仅扣佣金）。
- 基准① = 沪深300ETF(510300) 同期买入持有（全股）。
- 基准② = 建仓时股债比静态持有（60/40 不操作）——系统操作的超额以此为准。
- 记录 data/portfolio_track.jsonl（按日去重）；报告 portfolio_track_report.json。
- 建仓初期(<60日)样本无统计意义，报告如实标注。
"""
import json
from pathlib import Path

import backtest_lib as bl

ROOT = Path(__file__).resolve().parent.parent
PORT = ROOT / "data" / "portfolio.json"
TRACK = ROOT / "data" / "portfolio_track.jsonl"
REPORT = ROOT / "data" / "portfolio_track_report.json"


def close_on_or_before(px: dict, day: str):
    cands = [d for d in px if d <= day]
    return (max(cands), px[max(cands)]) if cands else (None, None)


def main():
    pf = json.loads(PORT.read_text(encoding="utf-8"))
    se, be = pf["stock_etf"], pf["bond_etf"]
    s_px = bl.fetch_daily(se, ROOT / "data" / "etf_cache")
    b_px = bl.fetch_daily(be, ROOT / "data" / "etf_cache")
    today = max(s_px)
    hold = pf.get("holdings", {})
    cash = float(pf.get("cash", 0))
    total = (hold.get(se, 0) * s_px[today]
             + hold.get(be, 0) * b_px.get(today, 0)) + cash

    # 按日记录（同日覆盖）
    rows = {}
    if TRACK.exists():
        for line in TRACK.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["date"]] = r
    rows[today] = {"date": today, "total": round(total, 2), "cash": round(cash, 2),
                   "n_stock": hold.get(se, 0), "n_bond": hold.get(be, 0),
                   "px_stock": s_px[today], "px_bond": b_px.get(today)}
    TRACK.write_text("\n".join(json.dumps(r, ensure_ascii=False)
                               for _, r in sorted(rows.items())) + "\n",
                     encoding="utf-8")

    # 建仓基准日 = portfolio created_at 的日期（首个跟踪日）
    created = pf.get("created_at", "")[:10]
    first_date = min(rows) if rows else today
    base_date = first_date
    d0s, p0s = close_on_or_before(s_px, base_date)
    d0b, p0b = close_on_or_before(b_px, base_date)
    initial_total = rows[first_date]["total"] if first_date in rows else total

    # 基准②的初始配置：首个跟踪日的股债市值比
    r0 = rows[first_date]
    hv_s0 = r0["n_stock"] * (p0s or 0)
    hv_b0 = r0["n_bond"] * (p0b or 0)
    w_s = hv_s0 / max(hv_s0 + hv_b0, 1e-9)
    w_b = 1 - w_s
    base300 = p0s and (s_px[today] / p0s - 1) * 100 or 0.0
    base6040 = ((w_s * s_px[today] / p0s + w_b * b_px.get(today, p0b) / p0b) - 1) * 100 \
        if p0s and p0b else 0.0
    track_ret = (total / initial_total - 1) * 100 if initial_total else 0.0
    days = len(rows)
    short = days < 60
    out = {
        "created_at": created, "base_date": base_date, "last_date": today,
        "days_tracked": days, "initial_total": round(initial_total, 2),
        "total_now": round(total, 2),
        "track_ret_pct": round(track_ret, 3),
        "bench_300_ret_pct": round(base300, 3),
        "bench_6040_ret_pct": round(base6040, 3),
        "excess_vs_6040_pp": round(track_ret - base6040, 3),
        "excess_vs_300_pp": round(track_ret - base300, 3),
        "short_sample": short,
        "note": "建仓不足60日，数字无统计意义，仅记录机制" if short else "",
    }
    REPORT.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"跟踪盘 {base_date} 建仓 → {today}（{days}日）")
    print(f"  总资产 {initial_total:,.0f} → {total:,.0f}  累计 {track_ret:+.3f}%")
    print(f"  同期 沪深300ETF {base300:+.3f}% | 60/40不操作 {base6040:+.3f}%")
    print(f"  系统操作超额(vs 60/40): {out['excess_vs_6040_pp']:+.3f}pp")
    if short:
        print("  （不足60日，样本无意义，机制演示期）")
    print(f"已写 {REPORT.name}")


if __name__ == "__main__":
    main()
