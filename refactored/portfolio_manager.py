#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""持仓管理 + 明确买卖指令生成（系统下达指令，不把判断推回给用户）。

设计原则：
  - 风险偏好是**一个输入参数**（可承受最大回撤），不是让用户自己判断"人性"
  - 系统据此自动反推股票/债券目标比例（来自 23.6 年回测的回撤-配置曲线）
  - 每天给出无歧义指令：买/卖哪只 ETF、多少份额；或"今日无需操作"
  - 调仓触发：年度再平衡（回测最优）+ 偏离阈值（±5pp）双条件

回测依据（asset_allocation_backtest.py, 2003-2026, 已补股息）:
  100%股: CAGR +8.44% 回撤-71.7% | 60/40: +8.54% -42.6% | 50/50: +8.14% -35.0%
  债券: +3.61% -10.4%  → 年度再平衡优于阈值触发
"""
import json
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PF = ROOT / "data" / "portfolio.json"
DEFAULT_STOCK_ETF = "sh510300"    # 沪深300ETF
DEFAULT_BOND_ETF = "sh511010"     # 国债ETF
REBALANCE_TOL = 0.05              # 偏离 5 个百分点触发
LOT = 100                         # ETF 最小 100 份

# 回撤 → 目标股票比例（来自回测，分段线性插值）
DD_CURVE = [(0.10, 0.0), (0.35, 0.5), (0.426, 0.6), (0.717, 1.0)]

# 估值倾斜（已验证：沪深300 相对自身5年(1220日)均线 z-score，全样本扣基准 t=+3.04；
# 近十年偏弱，未全过四门槛，仅作配置层比例微调，非个股买卖倾向）
TILT_K = 0.15
TILT_LO, TILT_HI = 0.40, 0.80
MA_WIN = 1220


def compute_valuation_z() -> float | None:
    """沪深300 相对自身 5 年均线的标准化偏离。负=便宜，正=贵。

    读本地缓存（无网络）。缓存缺失/过短/损坏 → None（倾斜失效，退回中性目标，
    fail-open 不报错）。估值倾斜是慢变量，缓存滞后数日不影响结论。
    """
    import statistics as _st
    cf = ROOT / "data" / "style_index_cache_6000" / "sh000300_6000.json"
    if not cf.exists():
        return None
    try:
        raw = json.loads(cf.read_text(encoding="utf-8"))
        # 缓存可能是 {date:close} 或 [[date,close],...] 两种格式
        if isinstance(raw, dict):
            items = sorted(raw.items())
        else:
            items = sorted(raw)
        prices = [c for _, c in items]
    except Exception:
        return None
    if len(prices) < MA_WIN:
        return None
    win = prices[-MA_WIN:]
    mu, sd = _st.mean(win), _st.pstdev(win)
    if sd <= 0:
        return None
    return (prices[-1] - mu) / sd


def tilt_target(neutral: float, z: float | None) -> tuple:
    """在风险中性目标上叠加估值倾斜。

    返回 (effective_target, z_or_None)。便宜(z<0)→提高股比，贵(z>0)→降低，
    限幅 [TILT_LO, TILT_HI]。z=None → 原样返回中性目标。
    """
    if z is None:
        return neutral, None
    eff = min(TILT_HI, max(TILT_LO, neutral - TILT_K * z))
    return round(eff, 4), z


def target_from_max_drawdown(max_dd: float) -> float:
    """给定可承受最大回撤(正数, 如0.30=30%)，反推目标股票比例。"""
    pts = sorted(DD_CURVE)
    if max_dd <= pts[0][0]:
        return pts[0][1]
    if max_dd >= pts[-1][0]:
        return pts[-1][1]
    for (d0, w0), (d1, w1) in zip(pts, pts[1:]):
        if d0 <= max_dd <= d1:
            return round(w0 + (w1 - w0) * (max_dd - d0) / (d1 - d0), 2)
    return 0.6


def get_prices(codes: list) -> dict:
    """sina 实时价。返回 {code: price}。"""
    out = {}
    try:
        q = ",".join(codes)
        req = urllib.request.Request(
            "https://hq.sinajs.cn/list=" + q,
            headers={"Referer": "https://finance.sina.com.cn"})
        txt = urllib.request.urlopen(req, timeout=15).read().decode(
            "gbk", "replace")
        for line in txt.strip().split("\n"):
            if '="' not in line:
                continue
            code = line.split("hq_str_")[1].split("=")[0]
            body = line.split('"')[1]
            f = body.split(",")
            if len(f) > 3 and f[3]:
                out[code] = float(f[3])
    except Exception:
        pass
    return out


def load() -> dict:
    if PF.exists():
        try:
            return json.loads(PF.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save(pf: dict):
    PF.parent.mkdir(parents=True, exist_ok=True)
    PF.write_text(json.dumps(pf, ensure_ascii=False, indent=2),
                  encoding="utf-8")


def init(total_cash: float = 100000.0, max_drawdown: float = 0.43,
         stock_etf: str = DEFAULT_STOCK_ETF,
         bond_etf: str = DEFAULT_BOND_ETF, virtual: bool = True) -> dict:
    """按目标比例建仓（默认虚拟跟踪，可改为录入实际持仓）。"""
    target = target_from_max_drawdown(max_drawdown)
    prices = get_prices([stock_etf, bond_etf])
    ps, pb = prices.get(stock_etf), prices.get(bond_etf)
    if not ps or not pb:
        raise RuntimeError(f"取价失败: {stock_etf}={ps} {bond_etf}={pb}")
    sv = total_cash * target
    # ETF 最小 100 份；债券 ETF 单价高(如141元→1手1.4万)，取整余款记入现金，
    # 现金视为无风险资产(等同债券)，否则会虚高股票占比。
    s_shares = int(sv / ps / LOT) * LOT
    remain = total_cash - s_shares * ps
    b_shares = int(remain / pb / LOT) * LOT
    cash = round(remain - b_shares * pb, 2)
    pf = {
        "virtual": virtual,
        "stock_etf": stock_etf,
        "bond_etf": bond_etf,
        "target_stock_ratio": target,
        "max_drawdown_tolerance": max_drawdown,
        "holdings": {stock_etf: s_shares, bond_etf: b_shares},
        "cash": cash,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "last_rebalance": datetime.now().strftime("%Y-%m-%d"),
    }
    save(pf)
    return pf


def evaluate(pf: dict) -> dict:
    """当前估值 + 偏离。"""
    s, b = pf["stock_etf"], pf["bond_etf"]
    prices = get_prices([s, b])
    ps, pb = prices.get(s), prices.get(b)
    if not ps or not pb:
        return {"ok": False, "reason": "取价失败"}
    sv = pf["holdings"].get(s, 0) * ps
    bv = pf["holdings"].get(b, 0) * pb
    cash = float(pf.get("cash", 0.0))
    total = sv + bv + cash          # 现金=无风险，计入分母
    if total <= 0:
        return {"ok": False, "reason": "持仓为空"}
    w = sv / total
    neutral = pf["target_stock_ratio"]
    eff, z = tilt_target(neutral, compute_valuation_z())
    return {"ok": True, "stock_value": round(sv, 2),
            "bond_value": round(bv, 2), "cash": round(cash, 2),
            "total": round(total, 2),
            "stock_ratio": round(w, 4),
            "target": eff,
            "neutral_target": round(neutral, 4),
            "tilt_z": round(z, 3) if z is not None else None,
            "deviation": round(w - eff, 4),
            "prices": {s: ps, b: pb},
            "last_rebalance": pf.get("last_rebalance", "")}


def actions(pf: dict) -> list:
    """生成明确买卖指令（份数取整到 100 份）。"""
    ev = evaluate(pf)
    if not ev.get("ok"):
        return [f"⚠️ 持仓评估失败: {ev.get('reason')}"]
    dev = ev["deviation"]
    s, b = pf["stock_etf"], pf["bond_etf"]
    ps, pb = ev["prices"][s], ev["prices"][b]
    out = []
    out.append(f"当前: 股票{ev['stock_ratio']:.1%} / 债券{1-ev['stock_ratio']:.1%} "
               f"（目标 {ev['target']:.0%}/{1-ev['target']:.0%}，"
               f"偏离 {dev*100:+.1f}pp），总资产约 {ev['total']:,.0f} 元")
    z = ev.get("tilt_z")
    if z is not None:
        band = "便宜(低于5年线)→多配股" if z < 0 else "贵(高于5年线)→少配股"
        out.append(f"  估值倾斜: 沪深300 z={z:+.2f}（{band}），"
                   f"中性目标 {ev['neutral_target']:.0%} → 倾斜后 {ev['target']:.0%}")
    elif ev.get("neutral_target") is not None:
        out.append(f"  估值倾斜: 缓存不可用，退回中性目标 {ev['neutral_target']:.0%}")
    if abs(dev) < REBALANCE_TOL:
        need = ev["target"] * ev["total"] - ev["stock_value"]
        out.append(f"✅ 今日无需操作（偏离未达 {REBALANCE_TOL:.0%} 触发线）")
        if abs(need) > 1:
            out.append(f"   参考：距目标差额约 {abs(need):,.0f} 元")
        return out
    # 需要调仓：目标股票市值 - 当前股票市值
    target_sv = ev["target"] * ev["total"]
    delta = target_sv - ev["stock_value"]
    cash = float(pf.get("cash", 0.0))
    if delta > 0:
        # 买股票：资金来自现金 + 卖债券
        need = delta
        from_cash = min(cash, need)
        from_bond = need - from_cash
        b_shares = int(from_bond / pb / LOT) * LOT
        s_shares = int((from_cash + b_shares * pb) / ps / LOT) * LOT
        if s_shares <= 0:
            out.append(f"✅ 偏离已触发，但可动用资金不足买入 1 手 {s}"
                       f"（需 ≥{ps*LOT:,.0f}元），本次不操作")
            return out
        amt = s_shares * ps
        parts = [f"买入 {s} {s_shares} 份（≈{amt:,.0f}元）"]
        if b_shares > 0:
            parts.append(f"卖出 {b} {b_shares} 份（≈{b_shares*pb:,.0f}元）")
        if from_cash > 0:
            parts.append(f"动用现金 {from_cash:,.0f} 元")
        out.append("🔧 再平衡指令：" + "；".join(parts))
    else:
        # 卖股票：资金买入债券（不足 1 手则留现金）
        amt = -delta
        s_shares = int(amt / ps / LOT) * LOT
        if s_shares <= 0:
            out.append(f"✅ 偏离已触发，但卖出量不足 1 手 {s}，本次不操作")
            return out
        proceeds = s_shares * ps + cash
        b_shares = int(proceeds / pb / LOT) * LOT
        parts = [f"卖出 {s} {s_shares} 份（≈{s_shares*ps:,.0f}元）"]
        if b_shares > 0:
            parts.append(f"买入 {b} {b_shares} 份（≈{b_shares*pb:,.0f}元）")
        leftover = proceeds - b_shares * pb
        if leftover > 1:
            parts.append(f"剩余 {leftover:,.0f} 元留作现金")
        out.append("🔧 再平衡指令：" + "；".join(parts))
    return out


def main():
    import sys
    pf = load()
    if not pf or "--init" in sys.argv:
        cash = float(next((a.split("=")[1] for a in sys.argv
                           if a.startswith("--cash=")), 100000))
        mdd = float(next((a.split("=")[1] for a in sys.argv
                          if a.startswith("--maxdd=")), 0.43))
        pf = init(total_cash=cash, max_drawdown=mdd)
        print(f"已初始化持仓（虚拟跟踪）：总资金 {cash:,.0f}，"
              f"可承受回撤 {mdd:.0%} → 目标股票比例 {pf['target_stock_ratio']:.0%}")
        print("  持仓:", pf["holdings"])
    ev = evaluate(pf)
    print(json.dumps(ev, ensure_ascii=False, indent=1))
    print("\n指令:")
    for line in actions(pf):
        print("  " + line)


if __name__ == "__main__":
    main()
