#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk-forward 历史回放回测 —— 加速样本积累，破解"每天只攒 7 笔"的龟速困局。

背景（为什么要它）：
  实时回测（backtest_signals.py）只能每交易日攒约 7 笔买入信号，且只覆盖"当前这一种
  市场状态"。要凑齐统计判决所需样本得等几周，要覆盖牛/熊/震荡得等几个月。

做法：
  把公式打分链路在过去 N 个交易日上逐日重放。对每个"锚日" i（= 生产早盘的"上一交易日"）：
    - 仅用 klines[i-59 : i+1]（最近 60 根，与生产 _fetch_sina_kline(days=60) 一致）算技术指标
    - 大盘状态仅用上证指数截至锚日的 60 根（与生产 fetch_index_klines("sh000001", 60) 一致）
    - entry = close[锚日]（生产早盘锚定"上一交易日收盘"）
    - 预测日 = 锚日的下一个交易日；日内收益 = close[i+1]/entry-1；隔日 = close[i+2]/entry-1
  口径与 backtest_signals.py 完全一致，因此两者结论可交叉验证。

零未来函数（已逐项坐实）：
  - technical: compute_real_technicals(仅锚日及之前的 60 根 K 线)
  - sentiment: sentiment_from_technical(上述 technical) —— 无独立舆情 API，纯技术衍生
  - fundamental: _neutral_fundamental(stock) —— 行业基准中性值，与日期无关（生产同款）
  - sector: relative_strength_score(个股 mom20, 指数 mom20)，指数动量截至锚日
  - weights: get_market_adjusted_weights(锚日 regime)
  - 【刻意排除】低置信降权 _apply_low_confidence —— 它依赖 iteration_briefing.txt
    （由"未来"的复盘写入），用在历史回放里等于引入未来信息，故不施加。

统计诚实性（重要）：
  同一交易日内 20 只股票的收益受共同市场因子驱动，高度相关。若把每笔当独立观测做 t 检验，
  独立性假设被违反，t 值会被显著高估（假显著）。因此本脚本同时输出两个口径：
    - 逐笔 t：n = 信号笔数（偏乐观，仅供参考）
    - 按天 t：n = 交易日数，每天取买入组合等权收益作为 1 个观测（保守稳健，**主判据**）
  下结论以「按天 t」为准。

用法：
  python3 refactored/walkforward_backtest.py [--days 180] [--fetch 300]
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import statistics as st
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from data_providers import (  # noqa: E402
    compute_real_technicals,
    detect_market_status,
    relative_strength_score,
    sentiment_from_technical,
    _neutral_fundamental,
)
from predict_then_summarize import (  # noqa: E402
    ConfigManager,
    ScoringEngine,
    SignalGenerator,
)
from backtest_signals import compute_verdict, verdict_lines_from_dict  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")

# 与生产严格对齐的窗口长度
WIN = 60                     # 个股技术指标窗口：akshare_fallback._fetch_sina_kline(days=60)
INDEX_WIN = 60               # 大盘状态窗口：predict_then_summarize 用 fetch_index_klines(..., 60)
INDEX_SYMBOL = "sh000001"    # 生产用上证指数判 regime
BUY_SIGNALS = ("买入", "强烈买入")
SELL_SIGNALS = ("卖出", "强烈卖出")


# ──────────────────────── 行情取数（完整 OHLCV，双源） ────────────────────────

def _sym(code: str) -> str:
    if code.startswith(("sh", "sz")):
        return code
    return ("sh" if code[0] in "69" else "sz") + code.zfill(6)


def fetch_ohlcv(code: str, days: int = 300) -> list:
    """返回升序 [{day, open, high, low, close, volume}, ...]；新浪优先、腾讯兜底。"""
    sym = _sym(code)
    # 新浪
    try:
        url = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
               f"CN_MarketData.getKLineData?symbol={sym}&scale=240&ma=no&datalen={days}")
        req = urllib.request.Request(url, headers={"Referer": "https://finance.sina.com.cn"})
        raw = urllib.request.urlopen(req, timeout=20).read().decode("gbk", "replace")
        rows = json.loads(raw)
        out = [
            {
                "day": r["day"], "open": r["open"], "high": r["high"],
                "low": r["low"], "close": r["close"], "volume": r["volume"],
            }
            for r in rows if r.get("day") and float(r.get("close") or 0) > 0
        ]
        if len(out) >= WIN + 3:
            return sorted(out, key=lambda x: x["day"])
    except Exception:
        pass
    # 腾讯兜底：[date, open, close, high, low, volume]
    try:
        url = ("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
               f"?param={sym},day,,,{days},qfq")
        req = urllib.request.Request(url, headers={"Referer": "https://finance.qq.com"})
        raw = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "replace")
        node = json.loads(raw)["data"][sym]
        rows = node.get("qfqday") or node.get("day") or []
        out = [
            {
                "day": r[0], "open": r[1], "close": r[2],
                "high": r[3], "low": r[4], "volume": r[5],
            }
            for r in rows if r and float(r[2] or 0) > 0
        ]
        return sorted(out, key=lambda x: x["day"])
    except Exception:
        return []


# ──────────────────────── 统计工具 ────────────────────────

def _tstat(xs: list):
    n = len(xs)
    if n < 2:
        return {"n": n, "mean": (xs[0] if n == 1 else 0.0), "sd": 0.0, "t": 0.0,
                "significant": False}
    m = st.mean(xs)
    sd = st.stdev(xs)
    se = sd / math.sqrt(n) if sd else 0.0
    t = (m / se) if se else 0.0
    return {"n": n, "mean": round(m, 4), "sd": round(sd, 4), "t": round(t, 2),
            "significant": abs(t) > 2.0}


def _winrate(xs: list) -> float:
    return round(100.0 * sum(1 for x in xs if x > 0) / len(xs), 1) if xs else 0.0


def _ma(closes: list, n: int) -> float | None:
    """简单移动平均；窗口不足返回 None。"""
    if len(closes) < n:
        return None
    return sum(closes[-n:]) / n


def _index_down_trend(idx_window: list) -> bool:
    """上证 20/60DMA 下跌趋势闸门（与 predict_then_summarize._detect_index_trend_gate 同口径）。

    收盘 < 60DMA 且 20DMA < 60DMA 判 down_trend。
    """
    closes = [float(k["close"]) for k in idx_window if float(k.get("close") or 0) > 0]
    if len(closes) < 60:
        return False
    ma20 = _ma(closes, 20)
    ma60 = _ma(closes, 60)
    if ma20 is None or ma60 is None:
        return False
    return closes[-1] < ma60 and ma20 < ma60


def _summarize(rows: list, key: str) -> dict:
    xs = [r[key] for r in rows if r.get(key) is not None]
    s = _tstat(xs)
    s["win_rate_pct"] = _winrate(xs)
    s["mean_pct"] = s.pop("mean")
    s["sd_pct"] = s.pop("sd")
    return s


def _by_day_series(rows: list, key: str) -> list:
    """每个预测日取等权组合收益，作为 1 个独立观测（消除同日横截面相关）。"""
    bucket = {}
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        bucket.setdefault(r["pred_day"], []).append(v)
    return [st.mean(v) for _, v in sorted(bucket.items()) if v]


# ──────────────────────── 主回放逻辑 ────────────────────────

def run_walkforward(replay_days: int | None = None, fetch_days: int = 300,
                   gated: bool = False) -> dict:
    stocks = ConfigManager.get_core_stocks()
    scoring = ScoringEngine()
    siggen = SignalGenerator()

    print(f"【Walk-forward 历史回放】股票池 {len(stocks)} 只 | 取数 {fetch_days} 根 | "
          f"窗口 {WIN} 根（与生产一致）" + (" | [闸门版] 下跌趋势日买入→持有" if gated else ""))

    # 1) 指数：预先算好每个锚日的 market_status（避免逐股重算）
    idx_kl = fetch_ohlcv(INDEX_SYMBOL, fetch_days)
    if len(idx_kl) < INDEX_WIN + 3:
        raise RuntimeError(f"指数 K 线不足：{len(idx_kl)} 根")
    idx_days = [k["day"] for k in idx_kl]
    idx_close = {k["day"]: float(k["close"]) for k in idx_kl}

    mstat_by_day: dict[str, dict] = {}
    gate_by_day: dict[str, bool] = {}   # 上证 20/60DMA 下跌趋势闸门（预计算，--gated 用）
    for j in range(INDEX_WIN - 1, len(idx_kl)):
        mstat_by_day[idx_days[j]] = detect_market_status(idx_kl[j - INDEX_WIN + 1: j + 1])
        gate_by_day[idx_days[j]] = _index_down_trend(idx_kl[j - INDEX_WIN + 1: j + 1])

    # 预测日的指数涨跌（仅用于事后 regime 分层分析，不参与决策）
    idx_ret = {}
    for j in range(1, len(idx_days)):
        prev, cur = idx_days[j - 1], idx_days[j]
        if idx_close[prev]:
            idx_ret[cur] = idx_close[cur] / idx_close[prev] - 1

    print(f"  指数 {INDEX_SYMBOL}: {idx_days[0]} ~ {idx_days[-1]}（{len(idx_kl)} 根）")

    # 2) 逐股回放
    rows: list[dict] = []
    skipped = []
    for stock in stocks:
        kl = fetch_ohlcv(stock.symbol, fetch_days)
        if len(kl) < WIN + 3:
            skipped.append(f"{stock.name}({stock.symbol})<{len(kl)}根>")
            continue
        days = [k["day"] for k in kl]
        closes = [float(k["close"]) for k in kl]

        start = WIN - 1
        end = len(kl) - 3          # 需要 i+1（预测日）与 i+2（隔日）
        if replay_days:
            start = max(start, end - replay_days + 1)

        fundamental = _neutral_fundamental(stock)
        fund_s = scoring.calculate_fundamental_score(fundamental, stock.sector)

        n_this = 0
        for i in range(start, end + 1):
            anchor_day = days[i]
            # 大盘状态：取截至锚日（含）最近一个有 status 的指数交易日
            j = bisect.bisect_right(idx_days, anchor_day) - 1
            if j < 0:
                continue
            mstat = mstat_by_day.get(idx_days[j])
            if mstat is None:
                continue

            win = kl[i - WIN + 1: i + 1]
            technical = compute_real_technicals(win)
            sentiment = sentiment_from_technical(technical)

            tech_s = scoring.calculate_technical_score(technical)
            sent_s = scoring.calculate_sentiment_score(sentiment)
            sector_s = relative_strength_score(
                float(technical.get("momentum_20d", 0.0)),
                float(mstat.get("trend_strength", 0.0)),
            )
            weights = ConfigManager.get_market_adjusted_weights(mstat.get("regime", "range"))
            final = scoring.calculate_final_score(
                tech_s, fund_s, sent_s, sector_s, weights=weights
            )
            signal, _conf, _reasons = siggen.generate_signal(final, stock, technical)
            orig_signal = signal  # 闸门前的原始信号（用于 ungated 基线对照）

            # ── 路线 A 闸门（仅 --gated）：下跌趋势日把买入/强烈买入压成持有 ──
            gate_down = gate_by_day.get(idx_days[j], False)
            a_suppressed = False
            if gated and gate_down and signal in BUY_SIGNALS:
                signal = "持有"
                a_suppressed = True

            entry = closes[i]
            if entry <= 0:
                continue
            rows.append({
                "pred_day": days[i + 1],
                "anchor_day": anchor_day,
                "symbol": stock.symbol,
                "name": stock.name,
                "signal": signal,
                "orig_signal": orig_signal,
                "final_score": final,
                "regime": mstat.get("regime", "range"),
                "status": mstat.get("status", "ranging"),
                "gate_down": gate_down,
                "a_suppressed": a_suppressed,
                "ret_intra": round((closes[i + 1] / entry - 1) * 100, 4),
                "ret_next": round((closes[i + 2] / entry - 1) * 100, 4),
                "idx_ret_pct": round((idx_ret.get(days[i + 1]) or 0.0) * 100, 4),
            })
            n_this += 1
        print(f"  {stock.name}({stock.symbol}): 回放 {n_this} 个决策日")

    if skipped:
        print(f"  ⚠️ 跳过（K线不足）: {', '.join(skipped)}")
    if not rows:
        raise RuntimeError("回放无有效样本")

    # 3) 聚合
    all_days = sorted({r["pred_day"] for r in rows})
    buy = [r for r in rows if r["signal"] in BUY_SIGNALS]          # 闸门后实际放行
    buy_all = [r for r in rows if r["orig_signal"] in BUY_SIGNALS]  # 原始公式买入意图
    sell = [r for r in rows if r["signal"] in SELL_SIGNALS]
    hold = [r for r in rows if r["signal"] not in BUY_SIGNALS + SELL_SIGNALS]

    print("\n" + "=" * 66)
    print(f"回放区间: {all_days[0]} ~ {all_days[-1]}（{len(all_days)} 个交易日）")
    print(f"总打分次数: {len(rows)} | 买入(放行) {len(buy)} | 卖出 {len(sell)} | 持有 {len(hold)}")
    if gated:
        print(f"  （路线 A 闸门：原公式买入意图 {len(buy_all)} 笔 → "
              f"下跌趋势日压掉 {len(buy_all) - len(buy)} 笔 → 实际放行 {len(buy)} 笔）")
    print("=" * 66)

    def _line(label, group):
        for key, kn in (("ret_intra", "日内"), ("ret_next", "隔日")):
            s = _summarize(group, key)
            print(f"  {label}·{kn}: n={s['n']:>5} 均值={s['mean_pct']:+.3f}% "
                  f"胜率={s['win_rate_pct']:>5.1f}% sd={s['sd_pct']:.2f}% "
                  f"t={s['t']:+.2f} {'显著' if s['significant'] else '不显著'}")

    print("\n【逐笔口径（偏乐观：同日横截面相关，独立性可疑，仅参考）】")
    _line("买入", buy)
    if sell:
        _line("卖出", sell)
    _line("持有", hold)

    # 按天口径（主判据）
    buy_day_next = _by_day_series(buy, "ret_next")
    buy_day_intra = _by_day_series(buy, "ret_intra")
    print("\n【按天口径（主判据：每天买入组合等权收益作 1 个观测）】")
    for xs, kn in ((buy_day_intra, "日内"), (buy_day_next, "隔日")):
        s = _tstat(xs)
        print(f"  买入组合·{kn}: n={s['n']} 交易日 | 均值={s['mean']:+.3f}%/日 "
              f"| 胜日率={_winrate(xs):.1f}% | sd={s['sd']:.2f}% | t={s['t']:+.2f} "
              f"{'显著' if s['significant'] else '不显著'}")

    # 4) regime 分层（决策时可得的锚日 regime）
    print("\n【按锚日 regime 分层（决策时可得，非事后）· 买入组·隔日】")
    regime_stats = {}
    for rg in sorted({r["regime"] for r in buy}):
        g = [r for r in buy if r["regime"] == rg]
        s = _summarize(g, "ret_next")
        dayxs = _by_day_series(g, "ret_next")
        ds = _tstat(dayxs)
        regime_stats[rg] = {"per_trade": s, "per_day": ds}
        print(f"  {rg:<11}: 笔数={s['n']:>5} 均值={s['mean_pct']:+.3f}% "
              f"胜率={s['win_rate_pct']:>5.1f}% | 按天 n={ds['n']:>3} "
              f"均值={ds['mean']:+.3f}% t={ds['t']:+.2f} "
              f"{'显著' if ds['significant'] else '不显著'}")

    print("\n【按锚日 status 细分（决策时可得）· 买入组·隔日】")
    status_stats = {}
    for stt in sorted({r["status"] for r in buy}):
        g = [r for r in buy if r["status"] == stt]
        s = _summarize(g, "ret_next")
        dayxs = _by_day_series(g, "ret_next")
        ds = _tstat(dayxs)
        status_stats[stt] = {"per_trade": s, "per_day": ds}
        print(f"  {stt:<15}: 笔数={s['n']:>5} 均值={s['mean_pct']:+.3f}% "
              f"胜率={s['win_rate_pct']:>5.1f}% | 按天 n={ds['n']:>3} "
              f"均值={ds['mean']:+.3f}% t={ds['t']:+.2f} "
              f"{'显著' if ds['significant'] else '不显著'}")

    print("\n【按预测日大盘涨跌分层（事后信息，仅诊断 regime 依赖）· 买入组·隔日】")
    market_split = {}
    for lab, pred in (("大盘上涨日", lambda r: r["idx_ret_pct"] > 0),
                      ("大盘下跌日", lambda r: r["idx_ret_pct"] <= 0)):
        g = [r for r in buy if pred(r)]
        if not g:
            continue
        s = _summarize(g, "ret_next")
        dayxs = _by_day_series(g, "ret_next")
        ds = _tstat(dayxs)
        market_split[lab] = {"per_trade": s, "per_day": ds}
        print(f"  {lab}: 笔数={s['n']:>5} 均值={s['mean_pct']:+.3f}% "
              f"胜率={s['win_rate_pct']:>5.1f}% | 按天 n={ds['n']:>3} t={ds['t']:+.2f}")

    # 5) 判据（复用实时回测的淘汰线逻辑；主判据用按天口径）
    verdict_per_day = compute_verdict(buy_day_next, buy_day_intra, full_population=True)
    verdict_per_trade = compute_verdict(
        [r["ret_next"] for r in buy], [r["ret_intra"] for r in buy],
        full_population=True,
    )

    # 5b) 闸门版（路线 A 体系）统计：仅 --gated 时计算并展示
    gated_out = None
    if gated:
        suppressed = [r for r in rows if r.get("a_suppressed")]
        # 未闸门（原始公式）判据 —— 对照基线（用 orig_signal 还原真实买入意图）
        base_day_next = _by_day_series(buy_all, "ret_next")
        base_day_intra = _by_day_series(buy_all, "ret_intra")
        base_v_day = compute_verdict(base_day_next, base_day_intra, full_population=True)
        base_v_trade = compute_verdict(
            [r["ret_next"] for r in buy_all], [r["ret_intra"] for r in buy_all],
            full_population=True,
        )
        supp_next = [r["ret_next"] for r in suppressed]
        supp_day_next = _by_day_series(suppressed, "ret_next")
        print("\n" + "=" * 66)
        print("【路线 A 体系（闸门版）· 未闸门原始公式 vs A 放行对比】")
        print(f"  原公式买入意图 {len(buy_all)} 笔 → 闸门压掉 {len(suppressed)} 笔 "
              f"→ A 实际放行 {len(buy)} 笔")
        print("  ── 未闸门（原始公式）判据（对照基线）──")
        for ln in verdict_lines_from_dict(base_v_day):
            if ln.strip():
                print(ln)
        print("  ── A 体系判据（已放行买入）──")
        for ln in verdict_lines_from_dict(verdict_per_day):
            if ln.strip():
                print(ln)
        if supp_next:
            ss = _tstat(supp_day_next)
            print(f"  被压掉的买入（下跌趋势日下行暴露，A 帮你避掉）: "
                  f"笔数={len(suppressed)} | 隔日均值={st.mean(supp_next):+.3f}%/笔 "
                  f"| 按天 n={ss['n']} 均值={ss['mean']:+.3f}%/日 t={ss['t']:+.2f}")
        else:
            print("  被压掉的买入: 0 笔")
        gated_out = {
            "orig_buy_count": len(buy_all),
            "suppressed_count": len(suppressed),
            "a_buy_count": len(buy),
            "a_buy_per_day": _tstat(buy_day_next),
            "a_buy_per_trade": _summarize(buy, "ret_next"),
            "baseline_per_day": base_v_day,
            "baseline_per_trade": base_v_trade,
            "suppressed_next_day": (_tstat(supp_day_next) if supp_next else None),
            "suppressed_mean_per_trade": round(st.mean(supp_next), 4) if supp_next else None,
            "verdict_per_day": verdict_per_day,
            "verdict_per_trade": verdict_per_trade,
        }

    print("\n" + "=" * 66)
    print("【判据 · 按天口径（主）】")
    for ln in verdict_lines_from_dict(verdict_per_day):
        if ln.strip():
            print(ln)
    print("\n【判据 · 逐笔口径（参考，t 被高估）】")
    for ln in verdict_lines_from_dict(verdict_per_trade):
        if ln.strip():
            print(ln)

    out = {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "method": "walk_forward_replay",
        "alignment": {
            "stock_window": WIN,
            "index_window": INDEX_WIN,
            "index_symbol": INDEX_SYMBOL,
            "entry": "close(anchor_day) == 生产早盘的上一交易日收盘锚",
            "low_confidence_penalty_applied": False,
            "note": "口径与 backtest_signals.py 一致，可交叉验证；未来函数已逐项排除",
        },
        "range": {"start": all_days[0], "end": all_days[-1], "trading_days": len(all_days)},
        "counts": {"scored": len(rows), "buy": len(buy), "sell": len(sell), "hold": len(hold)},
        "per_trade": {
            "buy": {"intraday": _summarize(buy, "ret_intra"),
                    "next_day": _summarize(buy, "ret_next")},
            "sell": {"intraday": _summarize(sell, "ret_intra"),
                     "next_day": _summarize(sell, "ret_next")} if sell else None,
            "hold": {"intraday": _summarize(hold, "ret_intra"),
                     "next_day": _summarize(hold, "ret_next")},
        },
        "per_day": {
            "buy_portfolio_intraday": _tstat(buy_day_intra),
            "buy_portfolio_next_day": _tstat(buy_day_next),
        },
        "by_regime_next_day": regime_stats,
        "by_status_next_day": status_stats,
        "by_market_direction_next_day": market_split,
        "verdict_per_day": verdict_per_day,
        "verdict_per_trade": verdict_per_trade,
        "gated": gated_out,
    }
    path = os.path.join(DATA_DIR, "walkforward_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已写 data/walkforward_report.json")
    return out


def main():
    ap = argparse.ArgumentParser(description="Walk-forward 历史回放回测")
    ap.add_argument("--days", type=int, default=None,
                    help="每只股票回放最近多少个决策日（默认全部可用）")
    ap.add_argument("--fetch", type=int, default=300,
                    help="取多少根历史日K（默认 300）")
    ap.add_argument("--gated", action="store_true",
                    help="闸门版重放：下跌趋势日（上证20/60DMA空头）把买入压成持有，"
                         "直接算路线 A 体系的历史 edge")
    args = ap.parse_args()
    run_walkforward(replay_days=args.days, fetch_days=args.fetch, gated=args.gated)


if __name__ == "__main__":
    main()
