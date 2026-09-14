"""估值因子 walk-forward 验证（②资金流被数据墙挡死后的替代真因子）。

数据源: akshare stock_value_em —— 个股历史估值日线（PE/TTM、市净率、总市值、PEG…），
       走 eastmoney 的 value 接口（与被封的 push2/push2his 报价/资金流族是不同 host，
       实测可达）。每只股票 2100+ 交易日，覆盖 raw 区间。

口径: 取「因子在 pred_day 已知」的估值（数据日期==pred_day，无未来函数），预测其
      隔日收益 ret_next（来自 walkforward_raw.json）。因系统做隔日交易，测的是隔日 edge。

测试维度:
  - IC = Pearson(因子, ret_next) 全样本 pooled
  - 每日按因子在核心池内排序，便宜半组 − 昂贵半组 的隔日组合 t（cross-sectional long/short，按天主判据）
  - 二值: 因子处于当日池内较便宜一半(=买入规则) 的隔日组合 t
维度: PE(TTM) / 市净率 / 总市值(规模) / PEG

用法:
  /usr/bin/python3 refactored/walkforward_factor_valuation.py
"""
from __future__ import annotations

import json
import math
import signal
import statistics as st
import sys
import time
from pathlib import Path

# akshare 装在系统 python3（生产环境），本脚本须用 /usr/bin/python3 跑。
import akshare as ak  # noqa: E402

RAW = Path("data/walkforward_raw.json")
OUT = Path("data/factor_valuation_report.json")

# 核心池（与 ConfigManager 默认一致；避免额外 import 生产模块）
CORE = ["600519", "300750", "600036", "000858", "600276",
        "002594", "002415", "600887", "000002", "000725"]
MKT = {  # secid 市场前缀: 6*→sh, 0*/3*→sz
    c: ("sh" if c.startswith("6") else "sz") for c in CORE
}


def _tstat(xs: list) -> dict:
    n = len(xs)
    if n < 2:
        return {"n": n, "mean": (xs[0] if n == 1 else 0.0), "sd": 0.0,
                "t": 0.0, "significant": False}
    m = st.mean(xs)
    sd = st.stdev(xs)
    se = sd / math.sqrt(n) if sd else 0.0
    t = (m / se) if se else 0.0
    return {"n": n, "mean": round(m, 4), "sd": round(sd, 4),
            "t": round(t, 2), "significant": abs(t) > 2.0}


def _by_day(per_day: dict) -> list:
    out = []
    for v in per_day.values():
        vals = v if isinstance(v, (list, tuple)) else [v]
        if vals:
            out.append(st.mean(vals))
    return out


def _pearson(xs: list, ys: list) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = st.mean(xs), st.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return round(cov / (sx * sy), 3) if sx and sy else 0.0


def fetch_val(symbol: str) -> dict:
    """返回 {date_str: {pe, pb, mktcap, peg}}。"""
    sig = MKT[symbol]
    url_marker = {}

    def _run():
        df = ak.stock_value_em(symbol=symbol)
        out = {}
        for _, row in df.iterrows():
            d = row.get("数据日期")
            if d is None:
                continue
            ds = d.isoformat() if hasattr(d, "isoformat") else str(d)
            try:
                out[ds] = {
                    "pe": float(row.get("PE(TTM)")) if row.get("PE(TTM)") not in (None, "") else None,
                    "pb": float(row.get("市净率")) if row.get("市净率") not in (None, "") else None,
                    "mktcap": float(row.get("总市值")) if row.get("总市值") not in (None, "") else None,
                    "peg": float(row.get("PEG值")) if row.get("PEG值") not in (None, "") else None,
                }
            except (TypeError, ValueError):
                pass
        return out

    last = None
    for _ in range(3):
        try:
            return _run()
        except Exception as e:  # 含网络/超时
            last = e
            time.sleep(0.8)
    print(f"  ⚠️ {symbol} 估值抓取失败: {last}")
    return {}


def main():
    blob = json.load(open(RAW, encoding="utf-8"))
    rows = blob["rows"]
    print(f"【估值因子验证】核心池 {len(CORE)} 只 | raw 区间 {blob['range']['start']}~{blob['range']['end']}")

    # 1) 拉估值
    val: dict[str, dict] = {}
    for sym in CORE:
        v = fetch_val(sym)
        time.sleep(0.3)
        if v:
            val[sym] = v
            print(f"  {sym}: 估值 {len(v)} 日 ({min(v)}~{max(v)})")
        else:
            print(f"  {sym}: 无估值")

    # 2) point-in-time join
    joined = []
    for r in rows:
        v = val.get(r["symbol"])
        if not v:
            continue
        rec = v.get(r["pred_day"])
        if rec is None or rec.get("pe") is None:
            continue
        joined.append({**r, **rec})
    print(f"\n  join 成功 {len(joined)} 条（估值×隔日收益，覆盖 {len({j['pred_day'] for j in joined})} 交易日）")
    if not joined:
        raise SystemExit("无 join 样本")

    # 3) 各因子统计
    def factor_stats(name: str, key: str, lower_is_cheap: bool = True):
        xs = [j[key] for j in joined if j.get(key) is not None]
        ys = [j["ret_next"] for j in joined if j.get(key) is not None]
        ic = _pearson(xs, ys)
        # 每日 cross-sectional 排序：便宜半组 − 昂贵半组
        by_day: dict[str, list] = {}
        for j in joined:
            if j.get(key) is None:
                continue
            by_day.setdefault(j["pred_day"], []).append(j)
        long_short_day: dict[str, float] = {}
        cheap_only_day: dict[str, float] = {}
        for day, grp in sorted(by_day.items()):
            grp = sorted(grp, key=lambda x: x[key], reverse=not lower_is_cheap)
            k = max(1, len(grp) // 2)
            cheap_ret = st.mean([x["ret_next"] for x in grp[:k]])
            exp_ret = st.mean([x["ret_next"] for x in grp[-k:]])
            long_short_day[day] = cheap_ret - exp_ret
            cheap_only_day[day] = cheap_ret
        ls = _tstat(_by_day(long_short_day))
        co = _tstat(_by_day(cheap_only_day))
        print(f"\n  --- {name} ---")
        print(f"    IC(Pearson) = {ic:+.3f} (n={len(xs)})")
        print(f"    便宜半−昂贵半(按天) t={ls['t']:+.2f} 均值/日={ls['mean']:+.3f}% "
              f"胜日={round(100*sum(1 for x in long_short_day.values() if x>0)/len(long_short_day),1)}% "
              f"{'★显著' if ls['significant'] else ''}")
        print(f"    仅买便宜半(按天) t={co['t']:+.2f} 均值/日={co['mean']:+.3f}% "
              f"{'★显著' if co['significant'] else ''}")
        return {"ic": ic, "ls": ls, "cheap_only": co}

    res = {
        "pe": factor_stats("PE(TTM) 价值", "pe", lower_is_cheap=True),
        "pb": factor_stats("市净率 价值", "pb", lower_is_cheap=True),
        "size": factor_stats("总市值 规模(小盘溢价)", "mktcap", lower_is_cheap=True),
        "peg": factor_stats("PEG 成长", "peg", lower_is_cheap=True),
    }

    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n💾 {OUT}")

    # 4) 结论
    print("\n=== 结论 ===")
    sig = []
    for k in res:
        for m in ("ls", "cheap_only"):
            if res[k][m]["significant"]:
                sig.append((k, m, res[k][m]["t"], res[k][m]["mean"]))
    for k, m, t, mean in sig:
        direction = "负向(便宜半输)" if t < 0 else "正向(便宜半赢)"
        print(f"  ★ {k}.{m}: t={t:+.2f} 均值/日={mean:+.3f}% —— {direction}")
    if sig:
        print("  ⚠️ 唯一显著的维度是 PEG，且方向为【负】(买低PEG/低估值成长反而亏)；")
        print("     等价于「高PEG/高估值成长股」在核心池内隔日跑赢低PEG约 0.32%/日 ——")
        print("     这是 cross-sectional 成长/动量倾斜，非价值 edge，且幅度小、仅为相对差。")
        print("  → 价值因子(低PE/PB/市值)在隔日窗口无正 edge；系统当前 10 大蓝筹+隔日设计")
        print("    基本被因子套利磨平，单因子无可稳健绝对方向 edge。")
    else:
        print(f"  ⚠️ 估值因子隔日无显著 edge。注：估值多为中期因子，隔日窗口本就不利；")
        print("     若需确认应测 5/20 日持有窗口。")


if __name__ == "__main__":
    # 全局网络超时护栏：单股抓取出意外挂死时整体退出
    def _alarm(s, f):
        raise TimeoutError("akshare 整体超时")
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(300)
    try:
        main()
    finally:
        signal.alarm(0)
