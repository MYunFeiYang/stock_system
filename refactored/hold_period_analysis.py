#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""持有期 vs 亏损概率：任意时点买入，持 N 年后亏钱的概率。

用户诉求"回撤=0"(不想亏)。金融学上回撤必然存在(风险溢价)，但**亏钱概率**
可以用持有期降低。本脚本用 23.6 年真实数据量化这个关系，给出诚实的
"持有多久才敢说不会亏"。

口径: 沪深300(已补年均股息2%) 与 60/40再平衡组合，滚动窗口统计。
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "style_index_cache_6000"
OUT = ROOT / "data" / "hold_period_report.json"
DIV = 0.02          # 年均股息
DIV_D = (1 + DIV) ** (1 / 244) - 1
YEARS = (1, 3, 5, 8, 10)


def load_close(sym):
    cf = CACHE / f"{sym}_6000.json"
    k = json.loads(cf.read_text(encoding="utf-8"))
    return dict((d, c) for d, c in k)


def main():
    stock = load_close("sh000300")
    bond = load_close("sh000012")
    start = max(min(stock), min(bond))
    dates = sorted(d for d in stock if d >= start)
    print(f"样本 {dates[0]} ~ {dates[-1]} ({len(dates)/244:.1f} 年)", flush=True)

    # 日收益序列
    sret, bret = {}, {}
    for i in range(1, len(dates)):
        d0, d1 = dates[i - 1], dates[i]
        if d1 in stock and d0 in stock and stock[d0] > 0:
            sret[i] = stock[d1] / stock[d0] - 1 + DIV_D
        if d1 in bond and d0 in bond and bond[d0] > 0:
            bret[i] = bond[d1] / bond[d0] - 1

    def roll(w_stock, years):
        """w_stock: 股票权重; 年度再平衡(每年244日)。返回各窗口总收益列表。"""
        n = int(years * 244)
        rets = []
        for i in range(1, len(dates) - n):
            eq = 1.0
            ws, wb = w_stock, 1 - w_stock
            for j in range(i, i + n):
                eq *= (1 + ws * sret.get(j, 0.0) + wb * bret.get(j, 0.0))
                ws2 = ws * (1 + sret.get(j, 0.0))
                wb2 = wb * (1 + bret.get(j, 0.0))
                t = ws2 + wb2
                ws, wb = ws2 / t, wb2 / t
                if (j - i + 1) % 244 == 0:      # 年度再平衡
                    ws, wb = w_stock, 1 - w_stock
            rets.append((eq - 1) * 100)
        return rets

    print(f"\n{'持有期':>6} | {'组合':<12}{'亏损概率':>9}{'中位收益':>10}"
          f"{'最差':>9}{'最好':>10}", flush=True)
    out = {}
    for years in YEARS:
        out[str(years)] = {}
        for label, w in (("满仓股票", 1.0), ("60/40再平衡", 0.6)):
            rs = roll(w, years)
            if not rs:
                continue
            loss = sum(1 for r in rs if r < 0) / len(rs)
            stat = {"loss_prob": round(loss, 3),
                    "median": round(st.median(rs), 1),
                    "worst": round(min(rs), 1),
                    "best": round(max(rs), 1),
                    "n": len(rs)}
            out[str(years)][label] = stat
            print(f"{years:>5}年 | {label:<12}{loss:>8.1%}"
                  f"{stat['median']:>+9.1f}%{stat['worst']:>+8.1f}%"
                  f"{stat['best']:>+9.1f}%", flush=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n已写 {OUT.name}", flush=True)


if __name__ == "__main__":
    main()
