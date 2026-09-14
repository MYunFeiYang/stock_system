#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测：早盘信号真实盈亏（股神风控用）

目的：回答"这套早盘信号到底赚没赚钱"——不只看方向一致率，而是算真实 P&L。

方法：
  - 扫描 data/predictions_morning_*.json，每个交易日取最后一个完整预测(count>=20)
  - entry = 预测文件里的 current_price（系统锚定的昨收）
  - exit1 = 预测日 close（日内口径：昨收买→今日收盘）
  - exit2 = 预测日下一交易日 close（隔日口径：持有1日）
  - 历史行情：新浪日K线优先，腾讯兜底（零依赖 urllib）
  - regime：沪深300 当日涨跌（close/前收-1）分上涨/下跌市

信号 P&L 定义（可交易代理）：
  - 买入：pnl = return（做多）
  - 卖出：pnl = -return（回避下跌/做空）
  - 持有：pnl = 0（不动作，不计入方向命中分母）

用法：
  python3 refactored/backtest_signals.py
"""
import json
import math
import os
import re
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime
from statistics import mean, stdev

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")
FILE_RE = re.compile(r"predictions_morning_(\d{8})_(\d{6})\.json$")
_KCACHE = {}

# ── 股神淘汰判据 ───────────────────────────────────────────────
# 主判决指标 = 隔日买入组合 P&L（持有至下一交易日收盘）。
# 日内口径样本需求过大(≈83交易日)直接放弃，只跟隔日。
# 满 ELIM_N 笔时看 t 值：|t|>2 即统计显著。
#   - 均值<0 且显著 → 无 edge，淘汰（停推买卖倾向，降为纯数据 feed）
#   - 均值>0 且显著 → 有 edge，可进入大盘趋势过滤(A)优化
#   - 仍不显著 → 续观察至 60 笔
# 未满 → 报距判决还差多少笔 + 约多少交易日（按日均≈7.3笔买入估计）
ELIM_N = 40
T_THRESHOLD = 2.0          # 双尾 alpha=0.05
POWER_COEF = 2.80          # (1.96+0.84)^2，80% power 样本量系数
AVG_BUY_PER_DAY = 7.3      # 估算：每个交易日约 7.3 笔买入信号
A_LAUNCH_DATE = "2026-08-24"  # 路线A(大盘趋势过滤)上线日；此前无闸门，其后买入信号含闸门调整


def _sym(code: str) -> str:
    return ("sh" if code[0] in "69" else "sz") + code.zfill(6)


def _fetch_kline(code: str, need_days: int = 25):
    """返回 { 'YYYY-MM-DD': close(float) }，优先新浪，腾讯兜底。"""
    if code in _KCACHE:
        return _KCACHE[code]
    out = {}
    # 新浪
    try:
        sym = _sym(code)
        url = (f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
               f"CN_MarketData.getKLineData?symbol={sym}&scale=240&ma=5&datalen={need_days}")
        req = urllib.request.Request(url, headers={"Referer": "https://finance.sina.com.cn"})
        raw = urllib.request.urlopen(req, timeout=12).read().decode("gbk", "replace")
        for c in json.loads(raw):
            if c.get("day"):
                out[c["day"]] = float(c["close"])
        if out:
            _KCACHE[code] = out
            return out
    except Exception:
        pass
    # 腾讯兜底 [date, open, close, high, low, volume]
    try:
        sym = _sym(code)
        url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
               f"?param={sym},day,,,{need_days},qfq")
        req = urllib.request.Request(url, headers={"Referer": "https://finance.qq.com"})
        raw = urllib.request.urlopen(req, timeout=12).read().decode("utf-8", "replace")
        d = json.loads(raw)["data"][sym]
        node = d["qfqday"] if "qfqday" in d else d["day"]
        for c in node:
            out[c[0]] = float(c[2])
        if out:
            _KCACHE[code] = out
            return out
    except Exception:
        pass
    _KCACHE[code] = out
    return out


def _index_returns(need_days: int = 25):
    """沪深300 每日 return(close/前收-1)，返回 { 'YYYY-MM-DD': ret }。"""
    closes = {}
    try:
        sym = "sh000300"
        url = (f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
               f"CN_MarketData.getKLineData?symbol={sym}&scale=240&ma=5&datalen={need_days}")
        req = urllib.request.Request(url, headers={"Referer": "https://finance.sina.com.cn"})
        raw = urllib.request.urlopen(req, timeout=12).read().decode("gbk", "replace")
        for c in json.loads(raw):
            if c.get("day"):
                closes[c["day"]] = float(c["close"])
    except Exception:
        return {}
    days = sorted(closes.keys())
    out = {}
    for i, d in enumerate(days):
        if i > 0:
            out[d] = closes[d] / closes[days[i - 1]] - 1
    return out


def _load_daily_files():
    """返回 { 'YYYY-MM-DD': predictions_list }，每个交易日取最后一个完整预测。"""
    by_date = defaultdict(list)
    for fn in os.listdir(DATA_DIR):
        m = FILE_RE.match(fn)
        if not m:
            continue
        ymd, hms = m.group(1), m.group(2)
        try:
            with open(os.path.join(DATA_DIR, fn), encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            continue
        preds = doc.get("predictions") or []
        if len(preds) < 20:
            continue  # 跳过开发期测试(count=5)
        day = f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}"
        by_date[day].append((hms, preds))
    # 每个交易日取时间最晚的一个
    return {d: max(v, key=lambda x: x[0])[1] for d, v in by_date.items()}


def _agg(rows):
    buy = [r for r in rows if r["signal"] == "买入"]
    sell = [r for r in rows if r["signal"] == "卖出"]
    hold = [r for r in rows if r["signal"] == "持有"]

    def stats(g, sign_fn):
        if not g:
            return None
        rets = [r["ret"] for r in g]
        pnls = [sign_fn(r["ret"]) for r in g]
        wins = sum(1 for p in pnls if p > 0)
        return {
            "n": len(g),
            "mean_ret_pct": round(sum(rets) / len(rets) * 100, 2),
            "mean_pnl_pct": round(sum(pnls) / len(pnls) * 100, 2),
            "win_rate": round(wins / len(g) * 100, 1),
        }

    return {
        "buy": stats(buy, lambda x: x),
        "sell": stats(sell, lambda x: -x),
        "hold": {"n": len(hold)} if hold else None,
        "all_pnl_pct": round(
            sum((r["ret"] if r["signal"] == "买入" else (-r["ret"] if r["signal"] == "卖出" else 0))
                for r in rows) / len(rows) * 100, 2) if rows else None,
    }


def compute_verdict(buy_next_pcts, buy_intra_pcts, full_population=False):
    """计算股神淘汰判据（基于隔日买入组合 P&L 的 t 检验）。

    full_population=True 用于 walk-forward 完整历史回放：样本已一次性充足，
    不再"滚动累积"，故不显著时直接下"无 detectable edge"的确定性结论，
    而非"再观察至 60 笔"。

    返回 dict：n / mean_pct / std_pct / t / significant / remaining / verdict / full_population。
    """
    n = len(buy_next_pcts)
    if n < 2:
        return {
            "usable": False, "n": n,
            "metric": "隔日买入组合 P&L",
            "full_population": full_population,
            "verdict": "样本不足(<2笔)，暂无法判决",
        }
    m = mean(buy_next_pcts)
    sd = stdev(buy_next_pcts)
    se = sd / math.sqrt(n)
    t = m / se if se else 0.0
    sig = abs(t) > T_THRESHOLD
    remaining = max(0, ELIM_N - n)
    # 检出该效应所需样本(80% power, alpha=.05)：n = (1.96+0.84)^2 * sd^2 / m^2
    need = (POWER_COEF ** 2) * (sd ** 2) / (m ** 2) if abs(m) > 1e-9 else float("inf")
    # 完整回放或已累积到充足样本：不再用"还差多少笔"措辞
    sufficient = full_population or n >= 60
    v = {
        "usable": True,
        "metric": "隔日买入组合 P&L（持有至下一交易日收盘）",
        "n": n,
        "mean_pct": round(m, 3),
        "std_pct": round(sd, 2),
        "t": round(t, 2),
        "significant": sig,
        "elim_n": ELIM_N,
        "remaining": 0 if sufficient else remaining,
        "need_total": (None if sufficient else (round(need) if need != float("inf") else None)),
        "full_population": full_population,
        "verdict": "",
    }
    if n >= ELIM_N:
        if sig:
            v["verdict"] = (
                ("无edge·淘汰" if m < 0 else "有edge")
                + f"：隔日买入均值 {m:+.2f}% 且 |t|={abs(t):.2f}>2，"
                + ("判定无 edge — 建议停止推送买卖倾向、降为纯数据 feed"
                   if m < 0 else
                   "判定有 edge — 建议进入大盘趋势过滤(A)优化")
            )
        elif sufficient:
            src = ("完整历史 walk-forward 回放样本已充足"
                   if full_population else "样本已累积至 60 笔")
            v["verdict"] = (
                f"无 detectable edge：{src}(n={n})，隔日买入组合均值 {m:+.2f}% "
                f"但 |t|={abs(t):.2f}<=2 不显著；公式 P&L 由市场 beta 主导，"
                f"无可分离的股票选择 alpha"
            )
        else:
            v["verdict"] = (
                f"续观察：满 {ELIM_N} 笔仍不显著(|t|={abs(t):.2f}<=2)，"
                f"按规则可再观察至 60 笔"
            )
    else:
        est_days = round(remaining / AVG_BUY_PER_DAY) if remaining else 0
        if sig:
            v["verdict"] = (
                f"观察中(预警)：样本 {n}/{ELIM_N}，距判决还差 {remaining} 笔"
                + (f"（约 {est_days} 个交易日）" if est_days else "")
                + f"；但 |t|={abs(t):.2f}>2 已显著，当前点估计 {m:+.2f}% 显著"
                + ("为负" if m < 0 else "为正")
                + f"，按规则满 {ELIM_N} 笔正式判决"
            )
        else:
            v["verdict"] = (
                f"观察中：样本 {n}/{ELIM_N}，距判决还差 {remaining} 笔"
                + (f"（约 {est_days} 个交易日）" if est_days else "")
                + f"；当前点估计 {m:+.2f}% 且 |t|={abs(t):.2f} 不显著，统计上与0无差异"
            )
    return v


def _index_closes_history(symbol: str = "sh000001", lookback: int = 400) -> dict:
    """取上证(默认)日线收盘序列 {date: close}，用于反事实闸门重算。

    复用 fetch_index_klines（akshare 优先、新浪兜底）。取大窗口以覆盖历史预测日，
    失败返回空 dict（调用方降级为 gate=normal）。
    """
    try:
        from akshare_fallback import fetch_index_klines
        kls = fetch_index_klines(symbol, lookback)
        return {k["day"]: float(k["close"]) for k in kls if k.get("close")}
    except Exception:
        return {}


def _index_gate_for_date(closes_map: dict, date: str) -> str:
    """按 date 当日及之前收盘算上证 20/60DMA 多空（与 predict_then_summarize._detect_index_trend_gate 同口径）。

    close < 60DMA 且 20DMA < 60DMA → 'down_trend'；否则 'normal'。
    数据不足(<60根) → 'normal'（fail-open）。
    """
    dates = sorted(d for d in closes_map if d <= date)
    if len(dates) < 60:
        return "normal"
    series = [closes_map[d] for d in dates]
    last = series[-1]
    ma20 = sum(series[-20:]) / 20.0
    ma60 = sum(series[-60:]) / 60.0
    return "down_trend" if (last < ma60 and ma20 < ma60) else "normal"


def _counterfactual_a(rows):
    """反事实：若 A(大盘下跌趋势闸门)从最早交易日即在线。

    公式本要买入(formula_signal='买入')的标的中，落在下跌趋势日(gate='down_trend')的
    会被压为持有——其隔日实际盈亏即 A 帮你避掉/少赚的部分。
    """
    would_suppress = [r for r in rows if r["formula_signal"] == "买入" and r["gate"] == "down_trend"]
    would_take = [r for r in rows if r["formula_signal"] == "买入" and r["gate"] != "down_trend"]

    def agg(g):
        if not g:
            return None
        rets = [r["ret_next"] for r in g]
        return {"n": len(g), "mean_next_pct": round(sum(rets) / len(rets), 2)}

    return {
        "total_buy_intent": len(would_suppress) + len(would_take),
        "suppressed_in_downtrend": agg(would_suppress),
        "taken_in_nondowntrend": agg(would_take),
    }


def compute_verdict_from_files():
    """静默版：读历史 predictions + 行情，返回 era-split + 反事实 verdict 结构。

    返回 dict:
      overall / pre_a / post_a: 各为 compute_verdict 的 verdict dict（或 None）
      counterfactual: 若 A 从最早交易日即在线，下跌趋势日被压买入的隔日盈亏
      a_launch: A 上线日
    注：混合判据(n=39)会被旧公式(36笔)主导而误导，故拆 pre_a/post_a 分段判决。
    """
    daily = _load_daily_files()
    idx_closes = _index_closes_history()  # 反事实闸门用；空→gate 全 normal

    rows = []
    for day in sorted(daily.keys()):
        era = "pre_a" if day < A_LAUNCH_DATE else "post_a"
        gate = _index_gate_for_date(idx_closes, day) if idx_closes else "normal"
        for p in daily[day]:
            code = p["stock"]["symbol"]
            entry = p.get("current_price")
            if not entry:
                continue
            kl = _fetch_kline(code)
            if day not in kl:
                continue
            days = sorted(kl)
            i = days.index(day)
            if i + 1 >= len(days):
                continue
            close_next = kl[days[i + 1]]
            fsig = p.get("formula_signal") or p.get("signal")  # 旧文件无 formula_signal→原始信号
            rows.append({
                "day": day, "era": era, "signal": p["signal"],
                "formula_signal": fsig, "gate": gate,
                "ret_next": (close_next / entry - 1) * 100,
            })

    def buys_by(pred):
        return [r["ret_next"] for r in rows if pred(r)]

    overall = compute_verdict(buys_by(lambda r: r["signal"] == "买入"), [])
    pre = compute_verdict(buys_by(lambda r: r["era"] == "pre_a" and r["signal"] == "买入"), [])
    post = compute_verdict(buys_by(lambda r: r["era"] == "post_a" and r["signal"] == "买入"), [])
    return {
        "overall": overall,
        "pre_a": pre if pre.get("usable") else None,
        "post_a": post if post.get("usable") else None,
        "counterfactual": _counterfactual_a(rows) if rows else None,
        "a_launch": A_LAUNCH_DATE,
    }


def _verdict_core_lines(v):
    """单个 verdict dict 的核心文本行（不含段标题）。供 verdict_lines_from_dict / verdict_lines_era 复用。"""
    if not v.get("usable"):
        return ["  样本不足(<2笔)，暂不可判"]
    if v.get("full_population"):
        elim_line = "  口径: 完整历史 walk-forward 回放（样本已充足，非实时滚动累积）"
    else:
        elim_line = (
            f"  淘汰线: 满 {v['elim_n']} 笔判决，当前还差 {v['remaining']} 笔"
            + (f"（约 {round(v['remaining']/AVG_BUY_PER_DAY)} 个交易日）" if v['remaining'] else "")
        )
    return [
        "  主指标: 隔日买入组合 P&L（持有至下一交易日收盘）",
        f"  样本 n={v['n']} | 均值={v['mean_pct']:+.2f}% | t={v['t']:+.2f} | "
        f"显著={'是' if v['significant'] else '否'}",
        elim_line,
        f"  判定: {v['verdict']}",
    ]


def verdict_lines_from_dict(v):
    """把单个 verdict dict 格式化成复盘报告文本块（list[str]）。保留给 walk-forward / run_backtest 用。"""
    if not v.get("usable"):
        return ["【信号回测判据】" + (v.get("verdict") or "样本不足，暂无法判决")]
    return ["", "【信号回测判据（股神淘汰线）】"] + _verdict_core_lines(v)


def verdict_lines_era(v):
    """era-split + 反事实 的判据文本块（list[str]）。供收盘复盘报告引用。"""
    a = v.get("a_launch", A_LAUNCH_DATE)
    lines = []
    overall = v.get("overall")
    if overall and overall.get("usable"):
        lines += ["", "【信号回测判据（股神淘汰线）· 全样本混合】"] + _verdict_core_lines(overall)
        lines.append("  ⚠️ 以上混合 A 前旧公式与 A 后体系，仅供总览；正式判决以分段为准")
    pre = v.get("pre_a")
    lines += ["", f"【A 前体系 · {a} 之前 · 旧公式(无闸门)】"]
    if pre:
        lines += _verdict_core_lines(pre)
        lines.append("  → 此为 A 上线前的旧公式表现，不代表 A 调整后体系")
    else:
        lines.append("  样本不足，暂不可判")
    post = v.get("post_a")
    lines += ["", f"【A 后体系 · {a} 及之后 · 含大盘趋势闸门】"]
    if post:
        lines += _verdict_core_lines(post)
    else:
        lines.append("  样本不足(<2笔)，暂不可判 — A 后买入信号稀少(下跌市被闸门压为持有)属正常")
    # 反事实
    cf = v.get("counterfactual")
    lines += ["", "【反事实 · 若 A 从最早交易日即在线】"]
    if not cf:
        lines.append("  无数据")
    else:
        tot = cf.get("total_buy_intent", 0)
        sup = cf.get("suppressed_in_downtrend")
        tak = cf.get("taken_in_nondowntrend")
        lines.append(f"  公式本要买入的信号共 {tot} 笔")
        if sup:
            lines.append(f"    - 下跌趋势日(闸门会压为持有): {sup['n']} 笔，隔日实际均值 {sup['mean_next_pct']:+.2f}%（即 A 帮你避掉的下行暴露）")
        else:
            lines.append("    - 下跌趋势日被压的买入: 0 笔")
        if tak:
            lines.append(f"    - 非下跌趋势日(闸门保留买入): {tak['n']} 笔，隔日实际均值 {tak['mean_next_pct']:+.2f}%")
        else:
            lines.append("    - 非下跌趋势日保留的买入: 0 笔")
        lines.append("  → 本窗口处下跌市，反事实主要说明 A 会'避跌'(压掉多数买入)；")
        lines.append("    因保留样本极少(仅数笔)，反事实不构成 A 有无 edge 的结论——")
        lines.append("    看上方【A 后体系】实际买入与【A 前】的方向反转更直观。")
    return lines


def run_backtest():
    daily = _load_daily_files()
    if not daily:
        print("无完整预测文件可回测")
        return
    idx = _index_returns()
    print(f"回测交易日: {', '.join(sorted(daily.keys()))}  (共 {len(daily)} 天)\n")

    all_rows = []
    up_rows, down_rows = [], []
    per_day = {}
    for day in sorted(daily.keys()):
        preds = daily[day]
        klines = {}
        miss = 0
        rows = []
        for p in preds:
            code = p["stock"]["symbol"]
            entry = p.get("current_price")
            if not entry:
                miss += 1
                continue
            kl = klines.get(code) or _fetch_kline(code)
            klines[code] = kl
            if day not in kl:
                miss += 1
                continue
            close_d = kl[day]
            days = sorted(kl.keys())
            i = days.index(day)
            close_next = kl[days[i + 1]] if i + 1 < len(days) else None
            ret_intra = close_d / entry - 1
            ret_next = (close_next / entry - 1) if close_next else None
            rows.append({
                "day": day, "code": code, "name": p["stock"]["name"],
                "signal": p["signal"], "final_score": p.get("final_score"),
                "entry": entry, "close_d": close_d, "ret": ret_intra,
                "ret_next": ret_next,
            })
        all_rows.extend(rows)
        per_day[day] = _agg(rows)
        ir = idx.get(day)
        if ir is not None:
            (up_rows if ir > 0 else down_rows).extend(rows)
        pending = (day >= datetime.now().strftime("%Y-%m-%d"))
        note = "（当日未收盘，跳过回测）" if (not rows and pending) else ""
        print(f"【{day}】 沪深300={('+%.2f' % (ir*100))+'%' if ir is not None else 'NA'} | "
              f"样本={len(rows)} 缺数={miss}{note}")
        a = per_day[day]
        print(f"   买入 {a['buy']}")
        print(f"   卖出 {a['sell']}")
        print(f"   持有 {a['hold']}")
        print(f"   等权组合P&L(日内) = {a['all_pnl_pct']}%\n")

    print("=" * 60)
    print("【按信号分层 · 全样本】")
    tot = _agg(all_rows)
    print(f"  买入 {tot['buy']}")
    print(f"  卖出 {tot['sell']}")
    print(f"  持有 {tot['hold']}")
    print(f"  等权组合P&L(日内) = {tot['all_pnl_pct']}%  | 样本 {len(all_rows)}")

    print("\n【按大盘regime · 日内】")
    print(f"  上涨市({len(up_rows)}): ", _agg(up_rows))
    print(f"  下跌市({len(down_rows)}): ", _agg(down_rows))

    # 隔日口径（仅对有 next close 的）
    buy_intra_pcts = [r["ret"] * 100 for r in all_rows if r["signal"] == "买入"]
    next_rows = [r for r in all_rows if r["ret_next"] is not None]
    buy_next_pcts = []
    if next_rows:
        for r in next_rows:
            r["ret"] = r["ret_next"]
        buy_next_pcts = [r["ret"] * 100 for r in next_rows if r["signal"] == "买入"]
        print("\n【隔日口径(持有至下一交易日收盘) · 全样本】")
        tn = _agg(next_rows)
        print(f"  买入 {tn['buy']}")
        print(f"  卖出 {tn['sell']}")
        print(f"  等权组合P&L(隔日) = {tn['all_pnl_pct']}%  | 样本 {len(next_rows)}")
        print(f"  上涨市: ", _agg([r for r in next_rows if (idx.get(r['day']) or 0) > 0]))
        print(f"  下跌市: ", _agg([r for r in next_rows if (idx.get(r['day']) or 0) <= 0]))

    # ── 股神淘汰判据（基于隔日买入组合 P&L 的 t 检验）──
    verdict = compute_verdict(buy_next_pcts, buy_intra_pcts)
    print("\n" + "=" * 60)
    print("【股神淘汰判据 · 隔日买入组合 P&L】")
    if verdict.get("usable"):
        print(f"  样本 n={verdict['n']} | 均值={verdict['mean_pct']:+.2f}% | "
              f"标准差={verdict['std_pct']:.2f}% | t={verdict['t']:+.2f} | "
              f"显著={'是' if verdict['significant'] else '否(|t|<=2)'}")
        print(f"  淘汰线: 满 {verdict['elim_n']} 笔判决（当前还差 {verdict['remaining']} 笔）")
    print(f"  判定: {verdict['verdict']}")

    # 落盘
    out = {
        "days": sorted(daily.keys()),
        "per_day": per_day,
        "overall": tot,
        "by_regime_intraday": {"up": _agg(up_rows), "down": _agg(down_rows)},
        "verdict": verdict,
    }
    with open(os.path.join(DATA_DIR, "backtest_report.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已写 data/backtest_report.json")
    return out


def verdict_section_lines():
    """供 daily_cycle_review 复盘报告引用的判据文本块（list[str]），静默计算。"""
    try:
        return verdict_lines_era(compute_verdict_from_files())
    except Exception as e:
        return [f"⚠️ 回测判据计算失败: {e}"]


if __name__ == "__main__":
    run_backtest()
