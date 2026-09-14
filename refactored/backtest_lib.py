#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""公共回测库：数据加载 / 组合模拟 / 统计检验，供验证与生产脚本复用。

统一口径（消除各脚本各写一份导致的口径漂移——2026-09-07 t 检验事故的教训）：
- STOCK_DIV 环境变量补股票股息（默认 2%/年，按 244 交易日摊入）
- 组合模拟 = 漂移 + 每 REBAL(244) 交易日再平衡，成本按净敞口双边扣
- t 检验 = 算术日收益差 + Newey-West 自相关调整（滞后阶 4*(T/100)^(2/9)）
"""
import json
import math
import os
import statistics as st
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "style_index_cache_6000"
COST = 0.002                                  # 再平衡双边 0.2%（净敞口口径）
DIV = float(os.environ.get("STOCK_DIV", "2")) / 100
DIV_D = (1 + DIV) ** (1 / 244) - 1            # 股票端日股息
REBAL = 244                                   # 年度（交易日）


def load(sym):
    """读指数缓存 {date: close}（data/style_index_cache_6000/{sym}_6000.json）。"""
    cf = CACHE / f"{sym}_6000.json"
    return dict((d, c) for d, c in json.loads(cf.read_text(encoding="utf-8")))


def fetch_daily(sym, cache_dir, datalen=6000):
    """sina 日线，本地缓存 {date: close}（ETF/现货通用）。"""
    cache_dir = Path(cache_dir)
    cf = cache_dir / f"{sym}.json"
    if cf.exists():
        return json.loads(cf.read_text(encoding="utf-8"))
    u = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
         f"CN_MarketData.getKLineData?symbol={sym}&scale=240&ma=no&datalen={datalen}")
    raw = urllib.request.urlopen(urllib.request.Request(
        u, headers={"Referer": "https://finance.sina.com.cn"}),
        timeout=20).read().decode("utf-8", "ignore")
    d = {x["day"]: float(x["close"]) for x in json.loads(raw)}
    cache_dir.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return d


def daily_returns(px, dates, div_daily=0.0):
    """给定价格字典与日期轴，返回日收益序列（首日 0）。缺价计 0。"""
    out = [0.0]
    for i in range(1, len(dates)):
        p0, p1 = px.get(dates[i - 1]), px.get(dates[i])
        out.append(p1 / p0 - 1 + div_daily if p0 else 0.0)
    return out


def portfolio_daily(rets, targets, rebal=REBAL, cost=COST):
    """N 资产组合日收益：漂移 + 定期再平衡（非再平衡日权重自由漂移）。

    rets: {资产名: 日收益序列}; targets: {资产名: 目标权重}
    成本 = Σ|Δw| × cost/2（净敞口双边）。
    """
    keys = list(targets)
    n = len(next(iter(rets.values())))
    w = [targets[k] for k in keys]
    out = [0.0] * n
    for i in range(1, n):
        out[i] = sum(w[k] * rets[k][i] for k in range(len(keys)))
        w = [w[k] * (1 + rets[k][i]) for k in range(len(keys))]
        tot = sum(w)
        w = [x / tot for x in w]
        if i % rebal == 0:
            turn = sum(abs(w[k] - targets[k_]) for k, k_ in enumerate(keys))
            out[i] -= turn * cost / 2
            w = [targets[k_] for k_ in keys]
    return out


def perf(rr, yrs):
    """年化 / 最大回撤 / 年化波动 / 夏普（sharpe = cagr/vol，口径同 timing_tilt）。"""
    g, peak, mdd = 1.0, 1.0, 0.0
    for x in rr[1:]:
        g *= (1 + x)
        peak = max(peak, g)
        mdd = min(mdd, g / peak - 1)
    cagr = (g ** (1 / yrs) - 1) * 100 if yrs else 0.0
    vol = st.pstdev([x * 100 for x in rr[1:]]) * (252 ** 0.5)
    return {"cagr": round(cagr, 2), "mdd": round(mdd * 100, 1),
            "vol": round(vol, 1), "sharpe": round(cagr / vol, 2) if vol else 0.0}


def tstat_nw(exc):
    """逐日超额 t 检验：返回 (普通t, Newey-West t, 滞后阶q)。样本<30 返回 (0,0,0)。"""
    N = len(exc)
    if N < 30:
        return 0.0, 0.0, 0
    mu = st.mean(exc)
    sd = (sum((x - mu) ** 2 for x in exc) / (N - 1)) ** 0.5
    t_ord = mu / (sd / math.sqrt(N))
    q = max(1, int(4 * (N / 100) ** (2 / 9)))
    g0 = sum((x - mu) ** 2 for x in exc) / N
    sac = g0
    for kk in range(1, q + 1):
        gk = sum((exc[t] - mu) * (exc[t - kk] - mu) for t in range(kk, N)) / N
        sac += 2 * gk * (1 - kk / (q + 1))
    return t_ord, mu / math.sqrt(sac / N), q


def zscore_series(prices, win=1220):
    """价格相对 win 日均线的 z-score 序列（窗口含当日；不足 win 日为 None）。"""
    out = []
    for i in range(len(prices)):
        if i < win:
            out.append(None)
            continue
        w = prices[i - win:i + 1]
        sd = st.pstdev(w)
        out.append((prices[i] - st.mean(w)) / sd if sd > 0 else None)
    return out
