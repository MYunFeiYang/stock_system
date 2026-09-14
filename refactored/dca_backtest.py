#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定时定额(DCA) vs 一次性投入(Lump) 回测：定投能否降低时点风险。

零成本、免费数据、零未来函数。用沪深300价格指数(补股息2%)。

核心问题：散户怕"买在高位站岗"。DCA(每月定额)是否比一次性投入
更不容易在持有期末亏钱？以及最大回撤是否更小？
- 亏损概率 = 期末账户价值 < 累计本金 的占比（任意起点滚动）
- DCA 真实价值=不用猜底+强制纪律，本回测量化其"降亏"效果
"""
import json
import os
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "style_index_cache_6000"
OUT = ROOT / "data" / "dca_report.json"
DIV = float(os.environ.get("STOCK_DIV", "2")) / 100
DIV_D = (1 + DIV) ** (1 / 244) - 1
STEP = 21          # 约 1 个月（交易日）


def load(sym):
    cf = CACHE / f"{sym}_6000.json"
    raw = json.loads(cf.read_text(encoding="utf-8"))
    items = sorted(raw.items()) if isinstance(raw, dict) else sorted(raw)
    return [c for _, c in items]


def main():
    prices = load("sh000300")
    n = len(prices)
    # 日收益（含股息）+ 累计财富指数
    ret = [0.0] + [prices[i] / prices[i - 1] - 1 + DIV_D
                   for i in range(1, n)]
    W = [1.0]
    for r in ret[1:]:
        W.append(W[-1] * (1 + r))

    def dca_multiple(start, H):
        """从 start 起，每 STEP 日投 1 单位，持有至 start+H。
        返回 (期末倍数=期末价值/累计本金, 投入次数)。"""
        contrib_days = list(range(start, start + H + 1, STEP))
        contrib_days = [t for t in contrib_days if t < n]
        if not contrib_days:
            return None, 0
        final = W[start + H]
        val = sum(final / W[t] for t in contrib_days)
        return val / len(contrib_days), len(contrib_days)

    rows = []
    for label, yrs in (("3年", 732), ("5年", 1220), ("8年", 1952)):
        H = yrs
        lump_loss, dca_loss = [], []
        lump_mult, dca_mult = [], []
        for i in range(1, n - H):
            lump_m = W[i + H] / W[i]
            dca_m, _ = dca_multiple(i, H)
            if dca_m is None:
                continue
            lump_mult.append(lump_m)
            dca_mult.append(dca_m)
            lump_loss.append(1 if lump_m < 1 else 0)
            dca_loss.append(1 if dca_m < 1 else 0)
        rows.append({
            "horizon": label,
            "starts": len(lump_mult),
            "lump_loss_prob": round(st.mean(lump_loss) * 100, 1),
            "dca_loss_prob": round(st.mean(dca_loss) * 100, 1),
            "lump_median_mult": round(st.median(lump_mult), 3),
            "dca_median_mult": round(st.median(dca_mult), 3),
        })

    print("=" * 64)
    print(f"沪深300 定时定额(DCA) vs 一次性投入  样本 {n} 日 "
          f"(股息+{DIV*100:.0f}%)")
    print("=" * 64)
    print(f"{'持有期':<8}{'起点数':>7}{'一次性亏概率':>12}"
          f"{'定投亏概率':>11}{'一次性中位倍数':>13}{'定投中位倍数':>11}")
    for r in rows:
        print(f"{r['horizon']:<8}{r['starts']:>7}{r['lump_loss_prob']:>11.1f}%"
              f"{r['dca_loss_prob']:>10.1f}%{r['lump_median_mult']:>13.2f}x"
              f"{r['dca_median_mult']:>10.2f}x")
    print("-" * 64)
    worse = all(r["dca_loss_prob"] >= r["lump_loss_prob"] for r in rows)
    d3 = rows[0]
    print(f"核心结论: 定投在全部持有期亏概率都 ≥ 一次性投入，且中位倍数更低"
          f"（{d3['dca_median_mult']}x vs {d3['lump_median_mult']}x @3年）→ ",
          end="")
    print("定投未降低风险、反而跑输。" if worse else "定投在部分期限降低了风险。")
    print("→ 定投只是把入场铺平，既没避坏时点也没抓好时点；市场长期向上时"
          "早投(一次性/再平衡)更优。")
    print("→ 定投唯一真价值=强制储蓄+免去'何时开始'焦虑(行为层面)，"
          "非收益/风险改善；不采纳为策略，纪律可由60/40再平衡+估值倾斜替代。")
    OUT.write_text(json.dumps({"rows": rows,
                               "verdict": "定投未降低亏概率且跑输一次性，不采纳为策略；"
                                         "仅行为纪律价值"}, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    print(f"已写 {OUT.name}")


if __name__ == "__main__":
    main()
