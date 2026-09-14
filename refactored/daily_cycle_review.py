#!/usr/bin/env python3
"""
日周期复盘（与 OpenClaw 取数一致）：

- reconcile：读取当日「早盘」预测落盘文件，收盘后再拉一次现价，用早盘价→现价
  的区间涨跌对照早盘信号，生成复盘报告（提高可检验性，而非再跑一套全量打分）。
- day_review：汇总当日已存在的 morning/afternoon/evening（及 weekly 若在同一天）
  的 predictions_*.json 元数据，并附带当日 reconcile；根据 reconcile_history.jsonl
  刷新 iteration_state.json 与 iteration_briefing.txt，供次日早盘报告引用（持续迭代）。

准确性建议（调度层面）：
- 早盘任务尽量放在开盘前；reconcile 放在收盘后足够晚（Cron 现为 18:10，晚于 18:00，依网页行情更新节奏可调），减少「价未更新 / 不代表全日收盘」的歧义。
- 仍依赖 OpenClaw Agent 拉价；若要与交易所官方收盘核对，可后续换数据源仅用于复盘价位。
- 复盘逻辑见 `reconcile_accuracy.py`（自适应中性带、强档更严、可选大盘超额）；调度侧可设 `RECONCILE_BENCHMARK_RETURN_PCT`。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from reconcile_accuracy import (
    adaptive_neutral_band_pct,
    benchmark_return_pct_for_reconcile,
    direction_match_with_tuning,
    expected_direction_from_signal,
    session_return_for_direction,
)

RECONCILE_HISTORY_FILENAME = "reconcile_history.jsonl"
ITERATION_STATE_FILENAME = "iteration_state.json"
ITERATION_BRIEFING_FILENAME = "iteration_briefing.txt"


def _root(base: Optional[str] = None) -> Path:
    if base:
        return Path(base)
    env = os.environ.get("STOCK_SYSTEM_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent


def _ymd() -> str:
    return datetime.now().strftime("%Y%m%d")


def _prediction_files_for_day(data_dir: Path, analysis_type: str, ymd: str) -> List[Path]:
    """文件名形如 predictions_morning_20260327_080530.json"""
    pat = f"predictions_{analysis_type}_*.json"
    out: List[Path] = []
    for p in sorted(data_dir.glob(pat)):
        if f"_{ymd}_" in p.name:
            out.append(p)
    return out


def _latest_prediction_for_day(data_dir: Path, analysis_type: str, ymd: str) -> Optional[Path]:
    files = _prediction_files_for_day(data_dir, analysis_type, ymd)
    return files[-1] if files else None


def _compact_symbols_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for r in rows:
        sym = str(r.get("symbol") or "").strip()
        if not sym:
            continue
        name = str(r.get("name") or "")
        if r.get("error"):
            out[sym] = {"name": name, "ok": False}
            continue
        out[sym] = {
            "name": name,
            "ok": True,
            "match": r.get("direction_match"),
            "signal": r.get("morning_signal"),
            "session_return_pct": r.get("session_return_pct"),
        }
    return out


def _append_reconcile_history(data_dir: Path, out_json: Dict[str, Any], reconcile_filename: str) -> None:
    path = data_dir / RECONCILE_HISTORY_FILENAME
    st = out_json.get("stats") or {}
    line = {
        "trade_date": out_json.get("trade_date"),
        "generated_at": out_json.get("generated_at"),
        "reconcile_file": reconcile_filename,
        "stats": {
            "hit_rate": st.get("hit_rate"),
            "with_prices": st.get("with_prices"),
            "direction_hits": st.get("direction_hits"),
            "direction_misses": st.get("direction_misses"),
            "fetch_errors": st.get("fetch_errors"),
            "rows": st.get("rows"),
            # 拆表口径（供滚动趋势与校准使用）
            "directional_samples": st.get("directional_samples"),
            "directional_hit_rate": st.get("directional_hit_rate"),
            "hold_hit_rate": st.get("hold_hit_rate"),
            "buy_signal_mean_return_pct": st.get("buy_signal_mean_return_pct"),
            "formula_directional_hit_rate": st.get("formula_directional_hit_rate"),
        },
        "symbols": _compact_symbols_from_rows(out_json.get("rows") or []),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def _check_strategy_health(data_dir: Path, window: int = 5, threshold: float = 0.5) -> Dict[str, Any]:
    """策略失效预警：连续 window 个交易日方向性一致率 < threshold 则标记 degraded 并企微告警。

    今天仅 1 天数据不会触发，需累积 window 天才有意义。返回当前状态 dict。
    新进入 degraded（上次非 degraded）才发企微，避免每次复盘重跑都推送。"""
    import os
    import subprocess
    path = data_dir / "strategy_status.json"
    records = _load_reconcile_history_records(data_dir, max_lines=200)
    recent = [r for r in records if (r.get("stats") or {}).get("directional_hit_rate") is not None]
    recent = recent[-window:]
    rates = [round(float(r["stats"]["directional_hit_rate"]), 4) for r in recent]
    status = "normal"
    since = None
    if len(recent) >= window and all(rt < threshold for rt in rates):
        status = "degraded"
        since = recent[0].get("trade_date")
    prev = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
    payload = {
        "status": status,
        "checked_at": datetime.now().isoformat(),
        "window": window,
        "threshold": threshold,
        "recent_directional_hit_rates": rates,
        "since": since,
    }
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[WARN] 写 strategy_status.json 失败: {e}")
    if status == "degraded" and prev.get("status") != "degraded":
        try:
            env = dict(os.environ)
            npm_bin = os.path.expanduser("~/.nvm/versions/node/v24.19.0/bin")
            if npm_bin not in env.get("PATH", ""):
                env["PATH"] = npm_bin + ":" + env.get("PATH", "")
            msg = (
                f"⚠️ 股票策略失效预警：连续 {window} 个交易日方向性一致率 < {threshold} "
                f"（最近 {len(rates)} 天: {rates}），策略疑似失效，已自动暂停买卖建议，建议观望。"
            )
            subprocess.run(
                ["openclaw", "message", "send", "--channel", "wecom", "--target", "thinkway", "--message", msg],
                env=env, capture_output=True, timeout=60,
            )
            print("📨 已发送策略失效企微告警")
        except Exception as e:
            print(f"[WARN] 策略失效企微告警发送失败: {e}")
    return payload


def _ensure_reconcile_history_seeded_from_disk(data_dir: Path) -> None:
    """若 jsonl 为空但已有历史 reconcile_*.json，按文件名顺序补写一行一条，便于立即启用滚动迭代。"""
    path = data_dir / RECONCILE_HISTORY_FILENAME
    if path.exists() and path.stat().st_size > 0:
        return
    for fp in sorted(data_dir.glob("reconcile_*.json")):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("kind") != "reconcile":
                continue
            _append_reconcile_history(data_dir, data, fp.name)
        except (OSError, json.JSONDecodeError, TypeError, KeyError):
            continue


def _load_reconcile_history_records(data_dir: Path, max_lines: int = 400) -> List[Dict[str, Any]]:
    path = data_dir / RECONCILE_HISTORY_FILENAME
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    out: List[Dict[str, Any]] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def _records_by_trade_date(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """同一交易日多次复盘时保留最后一次。"""
    by_d: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        d = rec.get("trade_date")
        if isinstance(d, str) and d:
            by_d[d] = rec
    return by_d


def _rolling_hit_rates(by_date: Dict[str, Dict[str, Any]], last_n: int, key: str = "hit_rate") -> Tuple[List[float], List[str]]:
    dates = sorted(by_date.keys())
    tail = dates[-last_n:] if len(dates) >= last_n else dates
    rates: List[float] = []
    for d in tail:
        st = (by_date[d].get("stats") or {})
        hr = st.get(key)
        if hr is not None:
            try:
                rates.append(float(hr))
            except (TypeError, ValueError):
                pass
    return rates, tail


def refresh_iteration_artifacts(data_dir: Path, trade_date_hint: str) -> Dict[str, Any]:
    """
    根据 reconcile_history.jsonl 生成持续迭代用的状态文件与次日早盘可读的简报。
    在 day_review 结束时调用；不调用大模型。
    """
    _ensure_reconcile_history_seeded_from_disk(data_dir)
    records = _load_reconcile_history_records(data_dir)
    by_date = _records_by_trade_date(records)
    dates_sorted = sorted(by_date.keys())

    last_20, _ = _rolling_hit_rates(by_date, 20)
    last_5, _ = _rolling_hit_rates(by_date, 5)
    prev_5_rates: List[float] = []
    if len(dates_sorted) >= 10:
        for d in dates_sorted[-10:-5]:
            st = (by_date[d].get("stats") or {})
            hr = st.get("hit_rate")
            if hr is not None:
                try:
                    prev_5_rates.append(float(hr))
                except (TypeError, ValueError):
                    pass

    mean_20 = round(sum(last_20) / len(last_20), 4) if last_20 else None
    mean_5 = round(sum(last_5) / len(last_5), 4) if last_5 else None
    mean_prev_5 = round(sum(prev_5_rates) / len(prev_5_rates), 4) if prev_5_rates else None

    # 方向性一致率滚动（剔除持有注水，才是真实预测能力）
    last_20_dir, _ = _rolling_hit_rates(by_date, 20, "directional_hit_rate")
    last_5_dir, _ = _rolling_hit_rates(by_date, 5, "directional_hit_rate")
    mean_20_dir = round(sum(last_20_dir) / len(last_20_dir), 4) if last_20_dir else None
    mean_5_dir = round(sum(last_5_dir) / len(last_5_dir), 4) if last_5_dir else None

    trend = "flat"
    trend_zh = "持平"
    if mean_5 is not None and mean_prev_5 is not None:
        if mean_5 > mean_prev_5 + 0.05:
            trend = "up"
            trend_zh = "改善"
        elif mean_5 < mean_prev_5 - 0.05:
            trend = "down"
            trend_zh = "回落"

    sym_events: Dict[str, List[Tuple[str, bool, str]]] = defaultdict(list)
    for d in dates_sorted:
        rec = by_date[d]
        for sym, info in (rec.get("symbols") or {}).items():
            if not isinstance(info, dict) or not info.get("ok", True):
                continue
            if info.get("match") is None:
                continue
            sig = str(info.get("signal") or "")
            sym_events[sym].append((d, bool(info["match"]), sig))

    attention: List[Dict[str, Any]] = []
    for sym, evs in sym_events.items():
        recent = evs[-5:]
        if len(recent) < 2:
            continue
        mism = sum(1 for _, m, _ in recent if not m)
        if mism >= 2:
            last_rec = (by_date[recent[-1][0]].get("symbols") or {}).get(sym) or {}
            name = str(last_rec.get("name") or sym)
            sig_order: List[str] = []
            for _, _, s in recent:
                if s and s not in sig_order:
                    sig_order.append(s)
            attention.append(
                {
                    "symbol": sym,
                    "name": name,
                    "recent_sessions": len(recent),
                    "recent_mismatch": mism,
                    "last_signals": sig_order[-3:],
                }
            )
    attention.sort(key=lambda x: (-x["recent_mismatch"], x["symbol"]))

    narrative: List[str] = []
    if mean_20 is not None:
        if len(last_20) == 1:
            narrative.append(
                f"当前仅有 1 次复盘记录，方向一致率约 {mean_20:.0%}；继续积累以便观察滚动趋势。"
            )
        else:
            narrative.append(f"近 {len(last_20)} 次复盘的方向一致率算术平均约 {mean_20:.0%}。")
    if mean_5 is not None and mean_prev_5 is not None:
        narrative.append(
            f"最近 5 次一致率 {mean_5:.0%}，对比此前 5 次 {mean_prev_5:.0%}，短期相对变化: {trend_zh}。"
        )
    elif mean_5 is not None and len(last_5) > 1:
        narrative.append(f"最近 {len(last_5)} 次复盘平均一致率约 {mean_5:.0%}。")

    # 方向性一致率（剔除持有注水）—— 真实预测能力口径
    if mean_20_dir is not None:
        if len(last_20_dir) == 1:
            narrative.append(
                f"当前仅 {len(last_20_dir)} 次含方向性信号（买卖）的复盘，方向性一致率约 {mean_20_dir:.0%}；"
                "继续积累以评估真实预测能力。"
            )
        else:
            narrative.append(
                f"近 {len(last_20_dir)} 次含方向性信号的复盘，方向性一致率（剔除持有注水）算术平均约 {mean_20_dir:.0%}。"
            )
    elif mean_20 is not None:
        narrative.append(
            "尚无任何买入/卖出方向性信号样本（系统仍偏保守）；新阈值（2026-08-11）将逐步产生方向性样本。"
        )

    if attention:
        top = attention[:5]
        narrative.append(
            "以下标的近期早盘方向与当日区间涨跌多次不一致，次日研判建议提高证据要求、核对信号与波动："
            + " ".join(f"{t['name']}({t['symbol']})" for t in top)
            + "。"
        )
    if not narrative:
        narrative.append("尚无足够历史复盘记录；积累数日 reconcile 后将自动生成趋势与标的提示。")
        if any(data_dir.glob("reconcile_*.json")):
            narrative.append(
                "提示: data/ 下已有 reconcile_*.json，但 reconcile_history.jsonl 尚无记录；"
                "请在更新本逻辑后至少成功执行一次 `reconcile`，之后每日会自动追加历史。"
            )

    state = {
        "updated_at": datetime.now().isoformat(),
        "as_of_trade_date": trade_date_hint,
        "window_effective_sessions": len(last_20),
        "mean_hit_rate_recent": mean_20,
        "last_5_mean_hit_rate": mean_5,
        "prev_5_mean_hit_rate": mean_prev_5,
        "mean_directional_hit_rate_recent": mean_20_dir,
        "last_5_mean_directional_hit_rate": mean_5_dir,
        "hit_rate_trend": trend,
        "hit_rate_trend_zh": trend_zh,
        "symbol_attention": attention[:15],
        "narrative_lines": narrative,
    }
    briefing_lines = [
        f"截至 {trade_date_hint} 晚间（由全日汇总生成，供下一交易日早盘对照）",
        "",
        *narrative,
        "",
    ]
    if attention:
        briefing_lines.append("【需加严复核的标的（近期多次方向不一致）】")
        for t in attention[:8]:
            briefing_lines.append(
                f"- {t['name']} ({t['symbol']}) 近{t['recent_sessions']}次中不一致 {t['recent_mismatch']} 次 "
                f"信号摘录: {', '.join(t['last_signals']) or '—'}"
            )
        briefing_lines.append("")

    briefing_lines.append(
        "说明: 一致率依赖 OpenClaw 拉价与早盘落盘价；仅作流程迭代参考，不构成投资建议。"
    )

    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / ITERATION_STATE_FILENAME, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    with open(data_dir / ITERATION_BRIEFING_FILENAME, "w", encoding="utf-8") as f:
        f.write("\n".join(briefing_lines))

    return state


@dataclass
class _StockLite:
    name: str
    symbol: str
    sector: str
    weight: float = 0.5


def run_reconcile(base_dir: Optional[str] = None, ymd: Optional[str] = None) -> int:
    """
    读取当日早盘 predictions 文件，对每只股票取交易日历史 OHLC（开盘→收盘）计算真实区间涨跌，
    与早盘信号方向对照，写入 data/reconcile_{ymd}_*.json 与 reports。
    采用交易日开/收（收盘后即固定的历史值），不受 cron 运行时刻影响，避免收盘后
    两次实时快照相等导致涨跌幅恒为 0 的虚假高一致率。
    """
    root = _root(base_dir)
    data_dir = root / "data"
    reports_dir = root / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    day = ymd or _ymd()

    morning_path = _latest_prediction_for_day(data_dir, "morning", day)
    if not morning_path:
        print(f"⚠️ 未找到当日早盘预测文件 predictions_morning_*_{day}_*.json，跳过复盘。")
        print("   请先成功跑过一次早盘分析，或将系统日期与文件内日期对齐。")
        return 2

    with open(morning_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)
    preds: List[Dict[str, Any]] = bundle.get("predictions") or []
    if not preds:
        print("❌ 早盘预测文件为空。")
        return 1

    # 延迟导入，避免无 OpenClaw 时影响其它子命令
    from akshare_fallback import fetch_daily_ohlc
    from predict_then_summarize import ConfigManager

    tuning = ConfigManager.get_accuracy_tuning()
    benchmark_ret = benchmark_return_pct_for_reconcile(day)

    rows: List[Dict[str, Any]] = []
    ok = 0
    for p in preds:
        stock_d = p.get("stock") or {}
        stock = _StockLite(
            name=str(stock_d.get("name", "")),
            symbol=str(stock_d.get("symbol", "")),
            sector=str(stock_d.get("sector", "")),
            weight=float(stock_d.get("weight", 0.5)),
        )
        morning_px = float(p.get("current_price") or 0)
        signal = str(p.get("signal") or "")
        exp = expected_direction_from_signal(signal)
        formula_signal = str(p.get("formula_signal") or "")
        fexp = expected_direction_from_signal(formula_signal) if formula_signal else 0
        morning_vol = abs(float(p.get("change_percent") or 0))
        band = adaptive_neutral_band_pct(morning_vol, tuning)

        # 用交易日历史 OHLC 计算真实区间涨跌（开盘→收盘），不依赖 cron 运行时刻。
        # 收盘后两次实时快照相等会让涨跌幅恒为 0（导致持有注水的虚假高分），
        # 而交易日的开/收是收盘后即固定的历史值，任何时刻都能取到。
        ohlc = fetch_daily_ohlc(stock.symbol, day)
        if ohlc is None:
            rows.append(
                {
                    "symbol": stock.symbol,
                    "name": stock.name,
                    "morning_price": None,
                    "evening_price": None,
                    "session_return_pct": None,
                    "morning_signal": signal,
                    "expected_direction": exp,
                    "direction_match": None,
                    "error": f"未取到 {stock.symbol} 交易日 {day} 的 OHLC 历史",
                }
            )
            continue
        open_px, close_px, _ = ohlc
        evening_px = close_px
        # morning_px 沿用早盘预测文件存的基线（昨收 prev_close，见第391行），
        # 与早盘「锚定上一交易日收盘」的预测口径一致；不再用当日开盘价覆盖，
        # 否则预测(相对昨收)与复盘(相对今开)口径错位，系统性污染方向一致率。
        if morning_px <= 0:
            morning_px = open_px  # 早盘文件缺基线时兜底用今开

        if morning_px <= 0:
            session_ret = 0.0
        else:
            session_ret = (close_px - morning_px) / morning_px * 100.0
        ret_for_match = session_return_for_direction(session_ret, benchmark_ret)
        matched, act, strong_band = direction_match_with_tuning(
            signal, ret_for_match, band, tuning
        )
        fmatched = None
        if formula_signal:
            fmatched, _, _ = direction_match_with_tuning(
                formula_signal, ret_for_match, band, tuning
            )
        if matched:
            ok += 1
        rows.append(
            {
                "symbol": stock.symbol,
                "name": stock.name,
                "morning_price": round(morning_px, 4),
                "evening_price": round(evening_px, 4),
                "session_return_pct": round(session_ret, 4),
                "benchmark_return_pct": benchmark_ret,
                "session_return_for_match_pct": round(ret_for_match, 4),
                "neutral_band_pct": band,
                "strong_band_pct": strong_band,
                "morning_signal": signal,
                "expected_direction": exp,
                "inferred_direction_from_session_return": act,
                "direction_match": matched,
                "formula_signal": formula_signal,
                "formula_expected_direction": fexp,
                "formula_direction_match": fmatched,
            }
        )

    total = len(rows)
    with_result = sum(1 for r in rows if r.get("session_return_pct") is not None)
    hit = sum(1 for r in rows if r.get("direction_match") is True)
    miss = sum(1 for r in rows if r.get("direction_match") is False)
    err_n = sum(1 for r in rows if r.get("error"))

    # ── 拆表：方向性信号（买入/卖出）vs 持有 ──
    # 持有信号 expected_direction=0，与“当日波动落在中性带内”天然对齐，注水严重；
    # 真正的预测能力只看方向性样本（买/卖）。
    dir_hit = dir_miss = hold_hit = hold_miss = 0
    fdir_hit = fdir_miss = 0  # A/B：纯公式路径方向性计数
    buy_rets: List[float] = []  # 买入信号股的当日区间收益（真实收益口径）
    sell_rets: List[float] = []  # 卖出信号股的当日区间收益（负=信号有效）
    hold_rets: List[float] = []  # 持有信号股的当日区间收益
    sell_hit = sell_miss = 0  # 卖出信号单独方向性计数
    for r in rows:
        if r.get("session_return_pct") is None or r.get("direction_match") is None:
            continue
        exp = int(r.get("expected_direction", 0) or 0)
        ok = bool(r.get("direction_match"))
        if exp != 0:  # 方向性信号
            if ok:
                dir_hit += 1
            else:
                dir_miss += 1
            if exp == 1:  # 买入信号
                buy_rets.append(float(r["session_return_pct"]))
            elif exp == -1:  # 卖出信号
                sell_rets.append(float(r["session_return_pct"]))
                if ok:
                    sell_hit += 1
                else:
                    sell_miss += 1
        else:  # 持有信号
            if ok:
                hold_hit += 1
            else:
                hold_miss += 1
            hold_rets.append(float(r["session_return_pct"]))
        # A/B 公式路径（纯公式信号的方向性对照）
        fexp = int(r.get("formula_expected_direction", 0) or 0)
        fok = r.get("formula_direction_match")
        if fexp != 0 and fok is not None:
            if fok:
                fdir_hit += 1
            else:
                fdir_miss += 1
    dir_samples = dir_hit + dir_miss
    directional_hit_rate = round(dir_hit / dir_samples, 4) if dir_samples else None
    hold_samples = hold_hit + hold_miss
    hold_hit_rate = round(hold_hit / hold_samples, 4) if hold_samples else None
    buy_mean_return = round(sum(buy_rets) / len(buy_rets), 4) if buy_rets else None
    sell_samples = sell_hit + sell_miss
    sell_hit_rate = round(sell_hit / sell_samples, 4) if sell_samples else None
    sell_mean_return = round(sum(sell_rets) / len(sell_rets), 4) if sell_rets else None
    hold_mean_return = round(sum(hold_rets) / len(hold_rets), 4) if hold_rets else None
    fdir_samples = fdir_hit + fdir_miss
    formula_directional_hit_rate = round(fdir_hit / fdir_samples, 4) if fdir_samples else None

    ts = datetime.now().strftime("%H%M%S")
    out_json = {
        "kind": "reconcile",
        "trade_date": day,
        "morning_source": str(morning_path),
        "generated_at": datetime.now().isoformat(),
        "reconcile_logic": {
            "adaptive_neutral_band": True,
            "strong_signal_stricter": True,
            "price_source": "trade_date_ohlc(open->close)",
            "benchmark_return_pct": benchmark_ret,
            "accuracy_tuning": tuning,
        },
        "stats": {
            "rows": total,
            "with_prices": with_result,
            "direction_hits": hit,
            "direction_misses": miss,
            "fetch_errors": err_n,
            "hit_rate": round(hit / with_result, 4) if with_result else None,
            # 拆表口径
            "directional_samples": dir_samples,
            "directional_hits": dir_hit,
            "directional_misses": dir_miss,
            "directional_hit_rate": directional_hit_rate,
            "hold_samples": hold_samples,
            "hold_hits": hold_hit,
            "hold_misses": hold_miss,
            "hold_hit_rate": hold_hit_rate,
            "buy_signal_mean_return_pct": buy_mean_return,
            "sell_signal_mean_return_pct": sell_mean_return,
            "hold_signal_mean_return_pct": hold_mean_return,
            "sell_samples": sell_samples,
            "sell_hit_rate": sell_hit_rate,
            "formula_directional_samples": fdir_samples,
            "formula_directional_hit_rate": formula_directional_hit_rate,
        },
        "rows": rows,
    }
    jpath = data_dir / f"reconcile_{day}_{ts}.json"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    _append_reconcile_history(data_dir, out_json, jpath.name)

    # ── 策略失效预警：连续 window 天方向性一致率 <0.5 标记 degraded + 企微告警 ──
    health = _check_strategy_health(data_dir)
    if health["status"] == "degraded":
        print(f"\n⚠️ 策略失效预警：连续 {health['window']} 日方向性一致率 {health['recent_directional_hit_rates']} "
              f"< {health['threshold']}，已自动暂停买卖建议，建议观望。")

    bench_note = (
        f"；大盘涨跌基准={benchmark_ret}%（用于超额收益判定，环境变量 RECONCILE_BENCHMARK_RETURN_PCT）"
        if benchmark_ret is not None
        else ""
    )
    lines = [
        f"【A股收盘复盘】交易日 {day}",
        f"早盘预测来源: {morning_path.name}",
        f"生成时间: {out_json['generated_at']}",
        "说明: 早盘价→收盘再拉价算区间涨跌；中性带按早盘|涨跌幅|自适应；"
        "强烈买入/卖出要求超过「强档带」幅度；普通买/卖与持有为三向对照。"
        + bench_note,
        "",
        f"统计: 共 {total} 条，有效比价 {with_result}，拉价失败 {err_n}",
        "",
        "【全量一致率（含持有，注水口径，仅供参考）】",
        f"  方向一致 {hit} / 有效 {with_result} = {out_json['stats']['hit_rate']!s}",
        "",
        "【方向性一致率（仅买入+卖出信号，真实预测能力口径）】",
        f"  一致 {dir_hit} / 样本 {dir_samples} = {directional_hit_rate!s}",
        f"  买入信号股当日平均区间收益: {buy_mean_return!s}%",
        f"  卖出信号股当日平均区间收益: {sell_mean_return!s}%  (负=卖出信号有效)",
        f"  持有信号股当日平均区间收益: {hold_mean_return!s}%",
        "",
        "【持有命中率（非预测能力，多数交易日波动<中性带天然判对）】",
        f"  一致 {hold_hit} / 样本 {hold_samples} = {hold_hit_rate!s}",
        "",
        "【LLM A/B 对照（纯公式 vs 公式+LLM 方向一致率）】",
        f"  纯公式: 一致 {fdir_hit} / 样本 {fdir_samples} = {formula_directional_hit_rate!s}",
        f"  公式+LLM: 一致 {dir_hit} / 样本 {dir_samples} = {directional_hit_rate!s}",
        "",
        "明细:",
    ]
    for r in rows:
        if r.get("error"):
            lines.append(f"- {r['name']}({r['symbol']}): 拉价失败 — {r['error'][:120]}")
        else:
            extra = ""
            if r.get("benchmark_return_pct") is not None:
                extra = f" | 判定用涨跌={r['session_return_for_match_pct']}%"
            lines.append(
                f"- {r['name']}({r['symbol']}): 早盘信号={r['morning_signal']} | "
                f"早盘价={r['morning_price']} → 现价={r['evening_price']} | "
                f"区间涨跌={r['session_return_pct']}%{extra} | "
                f"中性±{r['neutral_band_pct']}% | 方向一致={'是' if r['direction_match'] else '否'}"
            )

    # ── 信号回测判据（股神淘汰线）：收盘复盘后自动跑，判据写进复盘报告 ──
    # run_backtest() 重算全量回测并写 data/backtest_report.json；再取 verdict 附到本报告。
    try:
        from backtest_signals import run_backtest, verdict_lines_from_dict
        _bt = run_backtest()
        lines.extend(verdict_lines_from_dict(_bt.get("verdict")))
    except Exception as e:
        lines.append("")
        lines.append(f"【信号回测判据】计算跳过: {e}")

    text = "\n".join(lines)
    tpath = reports_dir / f"reconcile_report_{day}_{ts}.txt"
    with open(tpath, "w", encoding="utf-8") as f:
        f.write(text)

    print(text)
    print(f"\n💾 JSON: {jpath}")
    print(f"💾 报告: {tpath}")
    return 0


def run_day_review(base_dir: Optional[str] = None, ymd: Optional[str] = None) -> int:
    """汇总当日各档 predictions 文件（若存在），并引用最新 reconcile。"""
    from predict_then_summarize import ConfigManager
    from auto_calibration import run_auto_calibration

    root = _root(base_dir)
    data_dir = root / "data"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    day = ymd or _ymd()

    ConfigManager.reload_calibration()

    types_order = ("morning", "afternoon", "evening", "weekly")
    sections: List[str] = []
    meta: Dict[str, Any] = {"trade_date": day, "slots": {}}

    for t in types_order:
        path = _latest_prediction_for_day(data_dir, t, day)
        if not path:
            meta["slots"][t] = None
            continue
        with open(path, "r", encoding="utf-8") as f:
            b = json.load(f)
        preds = b.get("predictions") or []
        meta["slots"][t] = {
            "file": path.name,
            "count": len(preds),
            "timestamp": b.get("timestamp"),
        }
        th = ConfigManager.get_signal_thresholds()
        b_line, h_line = th["buy"], th["hold"]
        buy = sum(1 for p in preds if float(p.get("final_score", 0)) >= b_line)
        sell = sum(1 for p in preds if float(p.get("final_score", 0)) < h_line)
        sections.append(
            f"### {t}\n- 文件: {path.name}\n- 股票数: {len(preds)} | "
            f"买入档≥{b_line}: {buy} | 卖出档<{h_line}: {sell}"
        )

    # 最新 reconcile
    reconcile_files = sorted(data_dir.glob(f"reconcile_{day}_*.json"))
    rec_summary = None
    if reconcile_files:
        with open(reconcile_files[-1], "r", encoding="utf-8") as f:
            rec = json.load(f)
        st = rec.get("stats") or {}
        rec_summary = {
            "file": reconcile_files[-1].name,
            "hit_rate": st.get("hit_rate"),
            "hits": st.get("direction_hits"),
            "misses": st.get("direction_misses"),
        }
        sections.append(
            f"### 收盘复盘 (reconcile)\n- 文件: {reconcile_files[-1].name}\n"
            f"- 方向一致: {st.get('direction_hits')} / 有效样本 {st.get('with_prices')} "
            f"(一致率 {st.get('hit_rate')})"
        )

    iteration_state = refresh_iteration_artifacts(data_dir, day)
    cal_payload = run_auto_calibration(root)

    ts = datetime.now().strftime("%H%M%S")
    out_json = {
        "kind": "day_review",
        "trade_date": day,
        "generated_at": datetime.now().isoformat(),
        "meta": meta,
        "reconcile_summary": rec_summary,
        "iteration": {
            "mean_hit_rate_recent": iteration_state.get("mean_hit_rate_recent"),
            "last_5_mean_hit_rate": iteration_state.get("last_5_mean_hit_rate"),
            "prev_5_mean_hit_rate": iteration_state.get("prev_5_mean_hit_rate"),
            "hit_rate_trend_zh": iteration_state.get("hit_rate_trend_zh"),
            "symbol_attention_count": len(iteration_state.get("symbol_attention") or []),
            "briefing_file": ITERATION_BRIEFING_FILENAME,
            "state_file": ITERATION_STATE_FILENAME,
        },
        "auto_calibration": {
            "updated_at": cal_payload.get("updated_at"),
            "stats": cal_payload.get("stats"),
            "notes": cal_payload.get("notes"),
            "signal_thresholds": cal_payload.get("signal_thresholds"),
            "score_weights": cal_payload.get("score_weights"),
            "config_file": "config/calibration_overrides.json",
        },
    }
    jpath = data_dir / f"day_review_{day}_{ts}.json"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    iter_lines = [
        "### 持续迭代（已写入 data/iteration_briefing.txt，下一交易日早盘预测报告将自动引用）",
        "",
        *(iteration_state.get("narrative_lines") or []),
    ]
    att = iteration_state.get("symbol_attention") or []
    if att:
        iter_lines.append("")
        iter_lines.append("【近期方向多次不一致标的（摘要）】")
        for t in att[:6]:
            iter_lines.append(
                f"- {t.get('name')} ({t.get('symbol')}): 近{t.get('recent_sessions')}次中不一致{t.get('recent_mismatch')}次"
            )

    cal_notes = cal_payload.get("notes") or []
    iter_lines.append("")
    iter_lines.append("### 规则自校准（已写入 config/calibration_overrides.json，下一交易日早盘预测生效）")
    if cal_notes:
        iter_lines.extend(cal_notes)
    else:
        iter_lines.append("（无新说明）")
    iter_lines.append(
        f"当前阈值: buy={cal_payload.get('signal_thresholds', {}).get('buy')} "
        f"hold={cal_payload.get('signal_thresholds', {}).get('hold')} "
        f"sell={cal_payload.get('signal_thresholds', {}).get('sell')}"
    )

    lines = [
        f"【A股全日预测汇总】{day}",
        f"生成时间: {out_json['generated_at']}",
        "",
        "各时段落盘（若当日有跑过）:",
        "",
        "\n\n".join(sections) if sections else "（当日无任何 predictions_* 文件）",
        "",
        "\n".join(iter_lines),
        "",
        "说明: 汇总不调用大模型；迭代与规则自校准依据 reconcile_history.jsonl；阈值/权重见 config/calibration_overrides.json。",
    ]
    # ── 信号回测判据（股神淘汰线）并入汇总报告，随企微推送可见 ──
    try:
        from backtest_signals import verdict_section_lines
        lines.extend(verdict_section_lines())
    except Exception as e:
        lines.append("")
        lines.append(f"【信号回测判据】计算跳过: {e}")

    text = "\n".join(lines)
    tpath = reports_dir / f"day_review_report_{day}_{ts}.txt"
    with open(tpath, "w", encoding="utf-8") as f:
        f.write(text)

    print(text)
    print(f"\n💾 JSON: {jpath}")
    print(f"💾 报告: {tpath}")
    return 0


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="日周期复盘 / 全日汇总")
    p.add_argument(
        "command",
        choices=["reconcile", "day_review", "post_close"],
        help="post_close: 先 reconcile 再 day_review（与 openclaw_cron_analyzer post_close 一致）",
    )
    p.add_argument("--date", default=None, help="交易日 YYYYMMDD，默认今天")
    p.add_argument("--root", default=None, help="stock_system 根目录，默认本模块上级目录")
    a = p.parse_args()
    if a.command == "reconcile":
        raise SystemExit(run_reconcile(a.root, a.date))
    if a.command == "post_close":
        r1 = run_reconcile(a.root, a.date)
        r2 = run_day_review(a.root, a.date)
        # post_close 的核心产出是 day_review；reconcile 缺失早盘文件属于警告场景。
        raise SystemExit(r2)
    raise SystemExit(run_day_review(a.root, a.date))
