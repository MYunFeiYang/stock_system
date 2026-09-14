#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""股债配置 + 再平衡 —— 可执行的买卖框架（不依赖任何预测）。

为什么这是"能指导买卖"的框架：
  1. 不预测涨跌，只维持目标风险敞口（有经济学先验：风险分散 + 再平衡溢价）
  2. 再平衡天然产生明确的买卖指令：股票涨超目标 → 卖股买债；跌破 → 卖债买股
  3. 股债低相关，再平衡才真正有价值（纯股票间再平衡溢价≈0，已验证）
  4. 全部数据免费、官方指数零幸存者偏差、ETF 可执行

标的: 股票=沪深300(sh000300, ETF 510300) 或 4宽基等权
      债券=上证国债指数(sh000012, ETF 511010)
"""
import json
import os
import statistics as st
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "style_index_cache_6000"
OUT = ROOT / "data" / "asset_allocation_report.json"
COST = 0.0005      # ETF 单边 ≈0.05%（免印花税）


def fetch(sym):
    cf = CACHE / f"{sym}_6000.json"
    if cf.exists():
        return json.loads(cf.read_text(encoding="utf-8"))
    u = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
         f"CN_MarketData.getKLineData?symbol={sym}&scale=240&ma=no&datalen=6000")
    raw = urllib.request.urlopen(urllib.request.Request(
        u, headers={"User-Agent": "Mozilla/5.0",
                    "Referer": "https://finance.sina.com.cn"}),
        timeout=25).read().decode()
    d = json.loads(raw)
    out = [(r["day"], float(r["close"])) for r in d if r.get("close")]
    CACHE.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(out), encoding="utf-8")
    return out


def main():
    stock_syms = {"sh000300": "沪深300"}
    bond = dict(fetch("sh000012"))
    stock = dict(fetch("sh000300"))
    # 共同起始日
    start = max(min(stock), min(bond))
    dates = sorted({d for d in stock if d >= start and d in bond})
    print(f"覆盖 {dates[0]} ~ {dates[-1]} ({len(dates)/244:.1f} 年), "
          f"股票=沪深300 债券=上证国债指数", flush=True)

    # 股息修正: 指数为价格指数(不含股息), 债券指数为全收益 → 口径不对齐会系统性
    # 高估债券。用 A股年均股息率把股票端补成全收益口径(敏感性分析)。
    div = float(os.environ.get("STOCK_DIV", "0"))     # 年化股息率 %
    div_daily = (1 + div / 100) ** (1 / 244) - 1
    sret = {}
    bret = {}
    for i in range(1, len(dates)):
        d0, d1 = dates[i - 1], dates[i]
        if stock[d1] > 0 and stock[d0] > 0:
            sret[i] = stock[d1] / stock[d0] - 1 + div_daily
        if bond[d1] > 0 and bond[d0] > 0:
            bret[i] = bond[d1] / bond[d0] - 1
    if div:
        print(f"[股息修正] 股票端年化 +{div}% (价格指数→全收益口径)", flush=True)

    def sim(eq_w, mode, param=None):
        """eq_w: 股票目标权重; mode: hold/period/threshold"""
        ws, wb = eq_w, 1 - eq_w
        eq = 1.0
        eqs = []
        rets = []
        trades = []
        for i in range(1, len(dates)):
            ws *= (1 + sret.get(i, 0.0))
            wb *= (1 + bret.get(i, 0.0))
            tot = ws + wb
            ws, wb = ws / tot, wb / tot
            dret = tot - 1.0
            need = False
            if mode == "period" and i % param == 0:
                need = True
            elif mode == "threshold" and abs(ws - eq_w) >= param:
                need = True
            if need:
                turn = abs(ws - eq_w)
                eq *= (1 - turn * COST * 2)
                ws, wb = eq_w, 1 - eq_w
                trades.append((dates[i], round(ws * 100), round(turn * 100, 1)))
            eq *= (1 + dret)
            eqs.append(eq)
            rets.append(dret * 100)
        peak, mdd = 1.0, 0.0
        for e in eqs:
            peak = max(peak, e)
            mdd = min(mdd, e / peak - 1)
        years = len(dates) / 244
        cagr = (eq ** (1 / years) - 1) * 100
        vol = st.pstdev(rets) * (252 ** 0.5)
        sharpe = cagr / vol if vol else 0.0
        return {"cagr": round(cagr, 2), "mdd": round(mdd * 100, 1),
                "vol": round(vol, 1), "sharpe": round(sharpe, 2),
                "n_trades": len(trades), "trades": trades[-3:]}

    rows = []
    print(f"\n{'配置':<24}{'CAGR':>8}{'回撤':>9}{'波动':>8}{'夏普':>7}{'调仓':>7}",
          flush=True)
    cases = [
        ("100% 股票(基准)", 1.0, "hold", None),
        ("60/40 买入持有", 0.6, "hold", None),
        ("60/40 年度再平衡", 0.6, "period", 244),
        ("60/40 阈值±10%再平衡", 0.6, "threshold", 0.10),
        ("60/40 阈值±5%再平衡", 0.6, "threshold", 0.05),
        ("50/50 年度再平衡", 0.5, "period", 244),
        ("50/50 阈值±10%再平衡", 0.5, "threshold", 0.10),
        ("100% 债券", 0.0, "hold", None),
    ]
    for label, w, mode, p in cases:
        r = sim(w, mode, p)
        rows.append({"config": label, **r})
        print(f"{label:<24}{r['cagr']:>+7.2f}%{r['mdd']:>8.1f}%"
              f"{r['vol']:>7.1f}%{r['sharpe']:>7.2f}{r['n_trades']:>7}",
              flush=True)

    best = max(rows, key=lambda r: r["sharpe"])
    print(f"\n风险调整后最优: {best['config']} "
          f"(夏普{best['sharpe']}, CAGR{best['cagr']}%, 回撤{best['mdd']}%, "
          f"{best['n_trades']}次调仓)", flush=True)
    OUT.write_text(json.dumps({"start": dates[0], "end": dates[-1],
                               "stock": "沪深300(510300)",
                               "bond": "上证国债指数(511010)",
                               "rows": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"已写 {OUT.name}", flush=True)


if __name__ == "__main__":
    main()
