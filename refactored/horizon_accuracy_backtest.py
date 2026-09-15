#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预测周期准确性回测：用已有历史预测 + 股价历史，测 1/5/20 日周期的方向命中率与多空收益。

目的：检验"把预测周期从日频拉长到周/月频，能否提升准确性"这一假设。
数据：data/predictions_morning_*.json（历史每日预测，含 per-stock signal）
      data/history_cache/*.json（per-stock 日线收盘价，[date, close]）
方法：对每只 pred_date 的 买入/卖出 信号，取 entry=当日收盘，exit=后 H 日收盘，ret=exit/entry-1。
      买入命中=ret>0；卖出命中=ret<0。多空(LS)=买入均收益-卖出均收益，报 t 值。
不挑最优周期：1/5/20 三档并列报告，避免 multiple-testing 造假。
"""
import json
import glob
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load_hist():
    h = {}
    for f in glob.glob(str(DATA / "history_cache/*.json")):
        code = Path(f).stem
        rows = json.loads(Path(f).read_text())
        # rows: [[date, close], ...] 有序
        dates = [r[0] for r in rows]
        closes = [float(r[1]) for r in rows]
        h[code] = (dates, closes)
    return h


def load_preds():
    out = []
    for f in sorted(glob.glob(str(DATA / "predictions_morning_*.json"))):
        fn = Path(f).name
        # predictions_morning_YYYYMMDD_HHMMSS.json
        import re
        m = re.search(r"(\d{8})_(\d{6})", fn)
        if not m:
            continue
        pdate = m.group(1)
        o = json.loads(Path(f).read_text())
        preds = o.get("predictions", [])
        out.append((pdate, preds))
    return out


def norm_signal(s):
    if not s:
        return None
    s = str(s).strip()
    if s in ("买入", "buy", "bull", "bullish", "看多", "做多"):
        return "B"
    if s in ("卖出", "sell", "bear", "bearish", "看空", "做空"):
        return "S"
    if s in ("持有", "hold", "neutral", "中性"):
        return "H"
    return None


def main():
    hist = load_hist()
    preds = load_preds()
    # 收集 (signal, ret, pdate) per horizon
    rec = {1: [], 5: [], 20: []}
    sigvals = {}
    for pdate, plist in preds:
        pdate_fmt = f"{pdate[:4]}-{pdate[4:6]}-{pdate[6:8]}"
        for p in plist:
            raw = p.get("stock", "")
            code = str(raw.get("symbol", raw) if isinstance(raw, dict) else raw).zfill(6)
            sig = norm_signal(p.get("signal")) or norm_signal(p.get("formula_signal"))
            sigvals[sig] = sigvals.get(sig, 0) + 1
            if sig not in ("B", "S"):
                continue
            if code not in hist:
                continue
            dates, closes = hist[code]
            if pdate_fmt not in dates:
                continue
            i = dates.index(pdate_fmt)
            for H in (1, 5, 20):
                j = i + H
                if j < len(closes) and closes[i] > 0:
                    ret = closes[j] / closes[i] - 1.0
                    rec[H].append((sig, ret, pdate))
    print(f"历史预测文件: {len(preds)} 天")
    print(f"信号分布: {sigvals}")
    print()
    print(f"{'周期':>4} {'样本':>5} {'买入':>4} {'卖出':>4} {'方向命中':>8} {'买入均收益':>10} {'卖出均收益':>10} {'多空LS':>9} {'t':>7}")
    for H in (1, 5, 20):
        rows = rec[H]
        if not rows:
            print(f"{H:>4} {'-':>5}")
            continue
        bulls = [r for s, r, _ in rows if s == "B"]
        bears = [r for s, r, _ in rows if s == "S"]
        hits = sum(1 for s, r, _ in rows if (s == "B" and r > 0) or (s == "S" and r < 0))
        hr = hits / len(rows)
        mb = st.mean(bulls) if bulls else 0
        ms = st.mean(bears) if bears else 0
        ls = mb - ms
        n = len(bulls) + len(bears)
        sd = st.pstdev([mb - ms]) if False else None
        # t of LS: 用逐笔 (bull_ret - bear_ret) 配对不可行(数量不等)，改用两样本 t
        # 简化：t = LS / se，se = sqrt(var_b/n_b + var_s/n_s)
        var_b = st.pvariance(bulls) if len(bulls) > 1 else 0
        var_s = st.pvariance(bears) if len(bears) > 1 else 0
        se = (var_b / len(bulls) + var_s / len(bears)) ** 0.5 if (bulls and bears) else 0
        t = ls / se if se > 0 else 0
        print(f"{H:>4} {len(rows):>5} {len(bulls):>4} {len(bears):>4} {hr*100:>7.1f}% {mb*100:>9.2f}% {ms*100:>9.2f}% {ls*100:>8.2f}% {t:>7.2f}")
    print()
    print("说明: 方向命中率>50% 且多空LS t>2 才算'过线'。三档并列报告，不挑最优。")


if __name__ == "__main__":
    main()
