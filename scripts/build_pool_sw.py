"""生成申万一级行业覆盖的股票池（L1 工具化 · 扩池基础设施）。

为什么不用手写: 手写 150 个代码必然出错。改为从申万一级行业成分股按**权重**
(市值加权) 取每行业龙头，权重数据来自数据源，可复现、可校验。

数据源: akshare sw_index_first_info(31 个申万一级) + index_component_sw(成分股+权重)
        —— 均为非东财源，绕开被封的 push2 host。

用法(必须用系统 python3, akshare 装在那里):
  /usr/bin/python3 scripts/build_pool_sw.py [--per-industry 5]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import akshare as ak  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "config" / "stock_pool.json"
REPORT = ROOT / "data" / "pool_build_report.json"

KEEP = [  # 原核心池，必须保留（历史回测/复盘口径基于它们）
    "600519", "000858", "300750", "002594", "601012", "600036", "601398",
    "000001", "600276", "300760", "603259", "600887", "603288", "601888",
    "002415", "002475", "000725", "688981", "000002", "300059",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-industry", type=int, default=5)
    ap.add_argument("--out", default=str(POOL))
    args = ap.parse_args()

    inds = ak.sw_index_first_info()
    print(f"申万一级行业 {len(inds)} 个")

    picked: list[dict] = []
    seen: set[str] = set()
    per_ind = []

    for _, row in inds.iterrows():
        code = str(row["行业代码"]).split(".")[0]
        name = str(row["行业名称"])
        try:
            comp = ak.index_component_sw(symbol=code)
        except Exception as e:
            print(f"  ! {name}({code}) 成分股拉取失败: {type(e).__name__}")
            continue
        if comp is None or comp.empty:
            continue
        comp = comp.sort_values("最新权重", ascending=False)
        got = 0
        for _, c in comp.iterrows():
            sym = str(c["证券代码"]).zfill(6)
            if sym in seen:
                continue
            try:
                w = float(c["最新权重"])
            except Exception:
                w = 0.0
            seen.add(sym)
            picked.append({"name": str(c["证券名称"]), "symbol": sym,
                           "sector": name, "weight": round(min(1.0, w / 10.0), 2)})
            got += 1
            if got >= args.per_industry:
                break
        per_ind.append({"industry": name, "picked": got})
        print(f"  {name}: +{got}")
        time.sleep(0.15)      # 温和限速，避免被掐

    # 兜底：原核心池若未入选则补入（行业沿用申万映射，未知则标"核心"）
    sw_map = {p["symbol"]: p["sector"] for p in picked}
    added = []
    for s in KEEP:
        if s not in seen:
            picked.append({"name": s, "symbol": s,
                           "sector": sw_map.get(s, "核心"), "weight": 0.5})
            seen.add(s)
            added.append(s)
    if added:
        print(f"  补入原核心池未入选标的 {len(added)} 只")

    limits = {"morning": 150, "afternoon": 150, "evening": 150,
              "weekly": 150, "default": 150}
    out = {"analysis_limits": limits, "stocks": picked}
    Path(args.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    by_sec: dict[str, int] = {}
    for p in picked:
        by_sec[p["sector"]] = by_sec.get(p["sector"], 0) + 1
    print(f"\n生成 {len(picked)} 只，覆盖 {len(by_sec)} 个申万一级行业 → {args.out}")
    print("行业分布:", ", ".join(f"{k}{v}" for k, v in
                             sorted(by_sec.items(), key=lambda x: -x[1])[:12]), "...")

    REPORT.write_text(json.dumps(
        {"per_industry": args.per_industry, "total": len(picked),
         "industries": len(by_sec), "by_sector": by_sec,
         "kept_appended": added}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
