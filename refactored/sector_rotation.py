#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import pathlib as _pl
_orig_path_mkdir = _pl.Path.mkdir
def _broker_safe_mkdir(self, mode=0o777, parents=False, exist_ok=False):
    try:
        return _orig_path_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)
    except PermissionError:
        if exist_ok and self.exists():
            return None  # WorkBuddy broker 不支持 exist_ok 语义，已存在目录忽略
        raise
_pl.Path.mkdir = _broker_safe_mkdir
"""行业轮动预测（周频）—— 落盘 + 计分卡 + 数据驱动建议级别。

预测口径（不挑参数）：12 组参数（lookback 20/40/60/120 × TOP 2/3/5）各选当期
强势行业，按"入选票数"取共识 TOP3。行业动量回测背景（sector_rotation_backtest.py）：
扣 ETF 成本后超额年化 +10~23%，12/12 参数全正，OOS 12/12 同向。

核对口径：pred_date 后 5 个交易日，TOP3 行业等权收益 − 全行业等权 = 超额。
建议级别（写死防自欺）：滚动 12 次预测 t>+2 且 mean>0 → BUY_LEAN；
t<-2 → AVOID；其余 OBSERVE。动态计算，禁止硬编码。
"""
import datetime
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_relative_strength import load_pool, get_history  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "data" / "sector_rotation_log.jsonl"
SCORE = ROOT / "data" / "sector_rotation_scorecard.json"
LOOKBACKS = (20, 40, 60, 120)
TOPS = (2, 3, 5)
HOLD = 5           # 周频
ROLL_N = 12        # 滚动 12 周
MIN_N_FOR_ADVICE = 12
T_GATE = 2.0
ADVICE_TEXT = {"BUY_LEAN": "买入倾向", "AVOID": "回避提示", "OBSERVE": "观察"}


def build_sector_returns():
    stocks = load_pool()
    hist = {s["symbol"]: get_history(s["symbol"]) for s in stocks}
    sec_of = {s["symbol"]: s.get("sector", "未分类") for s in stocks}
    valid = {c: dict(h) for c, h in hist.items() if len(h) > 130}
    dates = sorted({d for v in valid.values() for d in v})
    idx = {d: i for i, d in enumerate(dates)}
    arrs = {c: {idx[d]: p for d, p in v.items()} for c, v in valid.items()}
    sectors = sorted({sec_of[c] for c in valid})
    sec_ret = {sec: {} for sec in sectors}
    for t in range(1, len(dates)):
        by = {}
        for c, a in arrs.items():
            if t in a and (t - 1) in a and a[t - 1] > 0:
                by.setdefault(sec_of[c], []).append(a[t] / a[t - 1] - 1)
        for sec, rs in by.items():
            sec_ret[sec][t] = st.mean(rs)
    return dates, idx, sec_of, sectors, sec_ret


def sector_perf(sec_ret, sectors, t, lookback):
    """过去 lookback 日各行业累计收益。"""
    perf = {}
    for sec in sectors:
        cum, ok = 1.0, True
        for tt in range(t - lookback + 1, t + 1):
            r = sec_ret[sec].get(tt)
            if r is None:
                ok = False
                break
            cum *= (1 + r)
        if ok:
            perf[sec] = cum - 1
    return perf


def consensus_prediction(sec_ret, sectors, t):
    """12 组参数共识：统计各行业入选 TOP 次数与平均排名。"""
    votes, ranks = {}, {}
    for lb in LOOKBACKS:
        perf = sector_perf(sec_ret, sectors, t, lb)
        if len(perf) < 5:
            continue
        ranked = sorted(perf, key=lambda s: -perf[s])
        for K in TOPS:
            for pos, sec in enumerate(ranked[:K]):
                votes[sec] = votes.get(sec, 0) + 1
                ranks.setdefault(sec, []).append(pos + 1)
    total_cells = len(LOOKBACKS) * len(TOPS)
    scored = sorted(votes.items(), key=lambda kv: (-kv[1],
                                                    st.mean(ranks[kv[0]])))
    out = [{"sector": s, "votes": v, "votes_pct": round(v / total_cells, 2),
            "avg_rank": round(st.mean(ranks[s]), 1)}
           for s, v in scored]
    return out, total_cells


def _tstat(xs):
    n = len(xs)
    if n < 2:
        return {"n": n, "mean": 0.0, "t": 0.0}
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return {"n": n, "mean": round(m, 4), "t": round(m / (sd / math.sqrt(n)), 2) if sd else 0.0}


def _advice_level(roll):
    if roll["n"] < MIN_N_FOR_ADVICE:
        return "OBSERVE"
    if roll["t"] > T_GATE and roll["mean"] > 0:
        return "BUY_LEAN"
    if roll["t"] < -T_GATE:
        return "AVOID"
    return "OBSERVE"


def check_predictions(dates, idx, sec_ret, sectors, records):
    """核对每条预测：未来 HOLD 日 TOP3 行业等权 − 全行业等权。"""
    checked = []
    for r in records:
        pd = r.get("pred_date")
        top = (r.get("top") or [])[:3]
        if not pd or pd not in idx:
            continue
        t = idx[pd]
        if t + HOLD >= len(dates):
            continue
        top_rets, all_rets = [], []
        for tt in range(t + 1, t + HOLD + 1):
            rs = [sec_ret[s].get(tt) for s in top if sec_ret[s].get(tt) is not None]
            if rs:
                top_rets.append(st.mean(rs))
            a = [sec_ret[s].get(tt) for s in sectors if sec_ret[s].get(tt) is not None]
            if a:
                all_rets.append(st.mean(a))
        if not top_rets or not all_rets:
            continue
        tm = st.mean(top_rets)
        am = st.mean(all_rets)
        hits = sum(1 for x, y in zip(top_rets, all_rets) if x > y)
        checked.append({
            "pred_date": pd,
            "top_ret_pct": round(tm * 100, 3),
            "bench_ret_pct": round(am * 100, 3),
            "excess_pct": round((tm - am) * 100, 3),
            "hit_rate": round(hits / len(top_rets), 2),
        })
    return checked


def update_scorecard(dates, idx, sec_ret, sectors):
    records = []
    if LOG.exists():
        for ln in LOG.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(ln))
            except Exception:
                pass
    checked = check_predictions(dates, idx, sec_ret, sectors, records)
    roll_ex = [c["excess_pct"] for c in checked[-ROLL_N:]]
    all_ex = [c["excess_pct"] for c in checked]
    roll = _tstat(roll_ex)
    level = _advice_level(roll)
    out = {
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_logged": len(records), "n_checked": len(checked),
        "all_time": _tstat(all_ex),
        "rolling": {**roll,
                    "hit_rate": round(sum(c["hit_rate"] for c in checked[-ROLL_N:])
                                      / len(roll_ex), 2) if roll_ex else None},
        "advice_level": level, "advice_text": ADVICE_TEXT[level],
        "recent": checked[-4:],
    }
    SCORE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def merge_log(recs, new):
    """按 pred_date 去重合并；实盘(live)优先于回填(backfill)。"""
    by_date = {}
    for r in list(recs) + list(new):
        d = r["pred_date"]
        old = by_date.get(d)
        if old is None or (old.get("type") == "backfill" and r.get("type") == "live"):
            by_date[d] = r
    return sorted(by_date.values(), key=lambda r: r["pred_date"])


def main():
    print("[1/4] 构建行业收益...")
    dates, idx, sec_of, sectors, sec_ret = build_sector_returns()

    print("[2/4] 当期共识预测...")
    t = len(dates) - 1
    ranked, total_cells = consensus_prediction(sec_ret, sectors, t)
    top3 = [r["sector"] for r in ranked[:3]]

    print("[3/4] 回填近 12 周历史预测 + 落盘...")
    old_recs = []
    if LOG.exists():
        for ln in LOG.read_text(encoding="utf-8").splitlines():
            try:
                old_recs.append(json.loads(ln))
            except Exception:
                pass
    new_recs = []
    backfill_from = max(121, len(dates) - 1 - 12 * HOLD)
    for tt in list(range(backfill_from, len(dates) - 1, HOLD)) + [len(dates) - 1]:
        if tt not in range(len(dates)):
            continue
        rk, _ = consensus_prediction(sec_ret, sectors, tt)
        if not rk:
            continue
        rec = {"pred_date": dates[tt],
               "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
               "top": [r["sector"] for r in rk[:3]],
               "type": "live" if tt == len(dates) - 1 else "backfill"}
        new_recs.append(rec)
    merged = merge_log(old_recs, new_recs)
    LOG.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in merged)
                   + "\n", encoding="utf-8")

    print("[4/4] 更新计分卡...")
    sc = update_scorecard(dates, idx, sec_ret, sectors)

    # 报告落盘（供推送读取）
    report = {
        "updated_at": sc["updated_at"],
        "last_date": dates[-1],
        "total_cells": total_cells,
        "prediction": ranked[:5],
        "top3": top3,
        "scorecard": {k: v for k, v in sc.items() if k != "recent"},
    }
    (ROOT / "data" / "sector_rotation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n当期({dates[-1]}) 行业轮动共识 TOP3: {top3}")
    for r in ranked[:5]:
        print(f"  {r['sector']}: 票数 {r['votes']}/{total_cells} "
              f"平均排名 {r['avg_rank']}")
    print(f"计分卡: n_checked={sc['n_checked']} rolling(t={sc['rolling']['t']}, "
          f"mean={sc['rolling']['mean']}%/周) 建议级别={sc['advice_level']}")


if __name__ == "__main__":
    main()
