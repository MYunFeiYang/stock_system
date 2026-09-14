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
"""行业中性相对强弱预测（日频 walk-forward）。

核心定位：把系统从「猜个股绝对方向」改为「预测行业内相对强弱」——后者是
有文献支撑（cross-sectional momentum / relative strength）且经统计功效分析
证明在扩池(150+只)后可达检测阈值的预测方式。

口径（零未来函数）：
  - 第 t 日收盘后，对每只股票算 mom = close[t]/close[t-60] - 1
  - 行业中性得分 score = mom - mean(同行业 mom)
  - 全池按 score 排序，做多前 1/10、做空后 1/10（理论 long-short）
  - 第 t+1 日收益 ret = close[t+1]/close[t] - 1
  - long-short 日收益 = mean(前1/10 ret) - mean(后1/10 ret)
  - walk-forward（每个 t 只用 t 及之前数据），逐日累积，对日 LS 收益序列算 t

输出：
  - data/rs_prediction_report.json：walk-forward 统计 + 最新一日 TOP 预测
  - 打印摘要
"""
import datetime
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from akshare_fallback import _fetch_sina_kline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "config" / "stock_pool.json"
CACHE = ROOT / "data" / "history_cache"
CACHE.mkdir(parents=True, exist_ok=True)

MOM_LOOKBACK = 60      # 动量回看窗口(交易日)
WARMUP = 120           # 起步需 120 根
TOP_FRAC = 0.10        # 做多前 1/10
DECILES = 10


def _sina_symbol(code: str) -> str:
    if code.startswith("6"):
        return "sh" + code
    if code[0] in "03":
        return "sz" + code
    return "sh" + code


def load_pool():
    d = json.load(open(POOL, encoding="utf-8"))
    return d["stocks"]


def get_history(code: str, days: int = 760, max_age_days: int = 3) -> list:
    """返回按日期升序的 [(date, close)]，缓存超过 max_age_days 天自动重拉。"""
    cache_f = CACHE / f"{code}.json"
    stale = None
    if cache_f.exists():
        try:
            stale = json.loads(cache_f.read_text(encoding="utf-8"))
            if stale:
                last = datetime.date.fromisoformat(stale[-1][0])
                if (datetime.date.today() - last).days <= max_age_days:
                    return stale
        except Exception:
            stale = None
    k = _fetch_sina_kline(_sina_symbol(code), days)
    if not k:
        return stale or []   # 重拉失败时退回旧缓存（fail-open）
    out = [(r["day"], float(r["close"])) for r in k if r.get("close")]
    cache_f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def build_cache(stocks, force=False):
    ok = 0
    for s in stocks:
        code = s["symbol"]
        cf = CACHE / f"{code}.json"
        if cf.exists() and not force:
            ok += 1
            continue
        h = get_history(code)
        if h:
            ok += 1
    return ok


def _tstat(xs: list) -> dict:
    n = len(xs)
    if n < 2:
        return {"n": n, "mean": 0.0, "sd": 0.0, "t": 0.0, "significant": False}
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(var)
    t = m / (sd / math.sqrt(n)) if sd else 0.0
    return {"n": n, "mean": round(m, 4), "sd": round(sd, 4),
            "t": round(t, 2), "significant": abs(t) > 2.0}


def predict_and_validate(stocks, hist_map: dict) -> dict:
    # hist_map: code -> [(date, close)]
    # 按交易日历（并集日期）对齐，杜绝不同长度序列索引错位
    valid = {s["symbol"]: dict(hist_map[s["symbol"]]) for s in stocks
             if len(hist_map.get(s["symbol"], [])) > MOM_LOOKBACK + 5}
    sec_of = {s["symbol"]: s.get("sector", "未分类") for s in stocks
              if s["symbol"] in valid}
    if len(valid) < 20:
        return {"error": f"有效股票不足: {len(valid)}"}

    dates = sorted({d for series in valid.values() for d in series})
    idx = {d: i for i, d in enumerate(dates)}
    # arr[sym][universe_index] = close
    arrs = {sym: {idx[d]: c for d, c in series.items()} for sym, series in valid.items()}
    name_of = {s["symbol"]: s["name"] for s in stocks}

    ls_rets, long_rets, bench_rets = [], [], []
    for t in range(MOM_LOOKBACK, len(dates) - 1):
        moms = {}
        for sym, a in arrs.items():
            if t in a and (t - MOM_LOOKBACK) in a and (t + 1) in a:
                c_prev = a[t - MOM_LOOKBACK]
                if c_prev > 0:
                    moms[sym] = a[t] / c_prev - 1.0
        if len(moms) < 20:
            continue
        # 行业中性
        by_sec = {}
        for sym, m in moms.items():
            by_sec.setdefault(sec_of[sym], []).append(m)
        sec_mean = {sec: sum(v) / len(v) for sec, v in by_sec.items()}
        scores = {sym: m - sec_mean[sec_of[sym]] for sym, m in moms.items()}
        ranked = sorted(scores.keys(), key=lambda s: scores[s])
        k = max(1, len(ranked) // DECILES)
        top, bot = ranked[-k:], ranked[:k]
        rt = {sym: arrs[sym][t + 1] / arrs[sym][t] - 1.0 for sym in moms}
        ls = st.mean(rt[s] for s in top) - st.mean(rt[s] for s in bot)
        ls_rets.append(ls * 100)
        long_rets.append(st.mean(rt[s] for s in top) * 100)
        bench_rets.append(st.mean(rt.values()) * 100)

    ls_stat = _tstat(ls_rets)
    lo_stat = _tstat(long_rets)
    be_stat = _tstat(bench_rets)
    excess = [a - b for a, b in zip(long_rets, bench_rets)]
    ex_stat = _tstat(excess)

    # 最新一日 TOP 预测
    last_t = len(dates) - 2
    moms = {}
    for sym, a in arrs.items():
        if last_t in a and (last_t - MOM_LOOKBACK) in a:
            c_prev = a[last_t - MOM_LOOKBACK]
            if c_prev > 0:
                moms[sym] = a[last_t] / c_prev - 1.0
    by_sec = {}
    for sym, m in moms.items():
        by_sec.setdefault(sec_of[sym], []).append(m)
    sec_mean = {sec: sum(v) / len(v) for sec, v in by_sec.items()}
    scores = {sym: m - sec_mean[sec_of[sym]] for sym, m in moms.items()}
    ranked = sorted(scores.keys(), key=lambda s: -scores[s])
    top_n = ranked[:15]
    current_pred = [{"symbol": s, "name": name_of.get(s, ""),
                     "sector": sec_of[s],
                     "mom_60d_pct": round(moms[s] * 100, 2),
                     "score": round(scores[s], 4)} for s in top_n]

    return {
        "n_stocks": len(valid),
        "n_days": len(ls_rets),
        "date_range": [dates[MOM_LOOKBACK], dates[-1]],
        "ls": ls_stat,
        "long_top_decile": lo_stat,
        "bench_equal_w": be_stat,
        "long_excess_vs_bench": ex_stat,
        "ls_annual_pct": round(ls_stat["mean"] * 252, 2),
        "long_annual_pct": round(lo_stat["mean"] * 252, 2),
        "current_prediction": current_pred,
        "last_date": dates[-1],
    }


def main():
    stocks = load_pool()
    print(f"[1/3] 加载池子: {len(stocks)} 只, 构建历史缓存...")
    build_cache(stocks)
    print(f"      缓存可用: {len(list(CACHE.glob('*.json')))} 只")

    print(f"[2/3] 读取历史并 walk-forward 验证...")
    hist_map = {}
    for s in stocks:
        h = get_history(s["symbol"])
        if h:
            hist_map[s["symbol"]] = h
    res = predict_and_validate(stocks, hist_map)

    print(f"[3/3] 结果:")
    print(f"  样本: {res['n_stocks']} 只 × {res['n_days']} 交易日")
    print(f"  多头前1/10(日收益): mean={res['long_top_decile']['mean']:+.4f}% "
          f"t={res['long_top_decile']['t']:+.2f} "
          f"年化≈{res['long_annual_pct']:+.1f}% "
          f"{'★显著' if res['long_top_decile']['significant'] else ''}")
    print(f"  等权基准(日收益):   mean={res['bench_equal_w']['mean']:+.4f}% "
          f"t={res['bench_equal_w']['t']:+.2f} 年化≈{res['bench_equal_w']['mean']*252:+.1f}%")
    print(f"  多头超额 vs 基准:    mean={res['long_excess_vs_bench']['mean']:+.4f}% "
          f"t={res['long_excess_vs_bench']['t']:+.2f} "
          f"{'★显著' if res['long_excess_vs_bench']['significant'] else ''}")
    print(f"  多空 LS(日收益):     mean={res['ls']['mean']:+.4f}% "
          f"t={res['ls']['t']:+.2f} 年化≈{res['ls_annual_pct']:+.1f}% "
          f"{'★显著' if res['ls']['significant'] else ''}")
    print(f"\n  最新一日({res['last_date']}) TOP 预测(行业中性相对强度):")
    for p in res["current_prediction"][:10]:
        print(f"    {p['symbol']} {p['name']:<8} [{p['sector']}] "
              f"60d动量{p['mom_60d_pct']:+.1f}% 中性分{p['score']:+.4f}")

    out = ROOT / "data" / "rs_prediction_report.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n  报告已写: {out}")

    # 预测落盘（供计分卡核对准确性）：按 pred_date 去重后 append
    log_f = ROOT / "data" / "rs_prediction_log.jsonl"
    pred_date = res.get("last_date")
    if pred_date and res.get("current_prediction"):
        recs = []
        if log_f.exists():
            for ln in log_f.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(ln)
                    if r.get("pred_date") != pred_date:
                        recs.append(r)
                except Exception:
                    pass
        recs.append({"pred_date": pred_date,
                     "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                     "top": [p["symbol"] for p in res["current_prediction"]]})
        log_f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs)
                         + "\n", encoding="utf-8")
        print(f"  预测落盘: {len(recs)} 条历史 -> {log_f.name}")

    # 顺手更新计分卡（预测准确性核对）
    try:
        from rs_scorecard import update_scorecard
        sc = update_scorecard()
        print(f"  计分卡: 已核对 {sc.get('n_checked', 0)} 次预测, "
              f"建议级别={sc.get('advice_level')}")
    except Exception as e:
        print(f"  计分卡更新失败(忽略): {e}")


if __name__ == "__main__":
    main()
