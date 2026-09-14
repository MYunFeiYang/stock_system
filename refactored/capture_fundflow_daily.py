"""每日主力资金流抓取（② 因子数据前向累积）。

数据源: eastmoney push2.eastmoney.com 实时排名接口（与被封的 push2his 历史接口
      是不同子域，实测可达）。一次调用返回全市场当日资金流，筛选核心池。

输出: data/fundflow_history.jsonl，每行 {date, symbol, name, flow_pct, flow_amt}
      —— point-in-time 累积，攒够样本后做 walk-forward 验证 / 接入实时信号。

用法:
  python3 refactored/capture_fundflow_daily.py          # 抓今日，append
  python3 refactored/capture_fundflow_daily.py --date 2026-09-05  # 指定日(补录)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_then_summarize import ConfigManager  # noqa: E402

OUT = Path("data/fundflow_history.jsonl")
HDR = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                     "AppleWebKit/537.36 Chrome/124 Safari/537.36",
       "Referer": "https://quote.eastmoney.com/", "Connection": "close"}

RANK_URL = (
    "https://push2.eastmoney.com/api/qt/clist/get"
    "?pn=1&pz=5000&po=1&np=1&fltt=2&invt=2&fid=f62"
    "&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
    "&fields=f12,f14,f62,f184,f66,f69,f72,f75,f78,f81,f84,f87"
    "&ut=b2884a393a59ad64002292a3e90d46a5"
)


def fetch_rank() -> list:
    last = None
    for _ in range(4):
        try:
            req = urllib.request.Request(RANK_URL, headers=HDR)
            raw = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "replace")
            obj = json.loads(raw)
            return (obj.get("data") or {}).get("diff") or []
        except Exception as e:
            last = e
            time.sleep(0.6)
    raise RuntimeError(f"资金流排名抓取失败: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().isoformat(),
                    help="抓取归属日(默认今天)，用于补录")
    args = ap.parse_args()
    d = args.date

    stocks = ConfigManager.get_core_stocks()
    want = {s.symbol for s in stocks}
    print(f"【资金流每日抓取】归属日 {d} | 核心池 {len(want)} 只")

    rows = fetch_rank()
    print(f"  排名接口返回 {len(rows)} 只全市场股票")
    got = 0
    out = []
    for r in rows:
        code = str(r.get("f12") or "").strip()
        if code not in want:
            continue
        flow_pct = r.get("f184")          # 主力净流入率 %
        flow_amt = r.get("f62")            # 主力净流入额(元)
        rec = {"date": d, "symbol": code,
               "name": str(r.get("f14") or ""),
               "flow_pct": float(flow_pct) if flow_pct not in (None, "") else None,
               "flow_amt": float(flow_amt) if flow_amt not in (None, "") else None}
        out.append(rec)
        got += 1
    print(f"  命中核心池 {got} 只")

    if not out:
        print("  ⚠️ 未命中任何核心池股票，检查符号映射")
        return

    # append（同日期去重：先读后写）
    OUT.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if OUT.exists():
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(json.loads(line))
    existing = [e for e in existing if e.get("date") != d]  # 覆盖当日
    existing.extend(out)
    with open(OUT, "w", encoding="utf-8") as f:
        for e in existing:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"  💾 {OUT}（累计 {len(existing)} 条）")
    for e in out[:5]:
        print(f"     {e['symbol']} {e['name']}: 主力净流入率={e['flow_pct']}%")


if __name__ == "__main__":
    main()
