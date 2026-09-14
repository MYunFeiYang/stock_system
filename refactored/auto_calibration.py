#!/usr/bin/env python3
"""
根据 reconcile_history.jsonl 的滚动结果，小步调整「信号阈值」与「综合分权重」，
写入 config/calibration_overrides.json；不修改模型，只调本地规则。

约束：单次日间调整幅度小、相对默认值有上下限；需满足 strong_buy > buy > hold > sell。

可调参数见 calibration_overrides.json 中 **auto_calibration**（与代码默认值合并）。
"""
from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _expected_direction(signal: str) -> int:
    if "强烈买入" in signal or signal == "买入":
        return 1
    if "强烈卖出" in signal or signal == "卖出":
        return -1
    return 0


DEFAULT_AUTO_CALIBRATION: Dict[str, Any] = {
    "decay_per_session": 0.88,
    "last_sessions": 20,
    "min_total_samples": 8,
    "min_bucket_samples": 3,
    "min_bucket_weight": 3.0,
    "max_thresh_drift": 0.35,
    "max_weight_drift": 0.06,
    "step_th": 0.05,
    "step_w": 0.02,
    "miss_rate_high": 0.45,
    "miss_rate_low": 0.2,
    "miss_imbalance_ratio": 1.4,
    "max_history_lines": 400,
}


def _sanitize_auto_tuning(raw: Dict[str, Any]) -> Dict[str, Any]:
    t = dict(DEFAULT_AUTO_CALIBRATION)
    for k, v in raw.items():
        if k not in t or v is None:
            continue
        if isinstance(t[k], int):
            t[k] = int(v)
        else:
            t[k] = float(v)
    t["decay_per_session"] = max(0.5, min(0.9999, float(t["decay_per_session"])))
    t["last_sessions"] = max(5, min(200, int(t["last_sessions"])))
    t["min_total_samples"] = max(5, min(500, int(t["min_total_samples"])))
    t["min_bucket_samples"] = max(2, min(50, int(t["min_bucket_samples"])))
    t["min_bucket_weight"] = max(1.0, min(50.0, float(t["min_bucket_weight"])))
    t["max_thresh_drift"] = max(0.05, min(1.0, float(t["max_thresh_drift"])))
    t["max_weight_drift"] = max(0.01, min(0.25, float(t["max_weight_drift"])))
    t["step_th"] = max(0.01, min(0.2, float(t["step_th"])))
    t["step_w"] = max(0.005, min(0.1, float(t["step_w"])))
    t["miss_rate_high"] = max(0.2, min(0.8, float(t["miss_rate_high"])))
    t["miss_rate_low"] = max(0.05, min(0.5, float(t["miss_rate_low"])))
    if float(t["miss_rate_low"]) >= float(t["miss_rate_high"]):
        t["miss_rate_low"] = max(
            0.05, min(0.25, float(t["miss_rate_high"]) - 0.08)
        )
    t["miss_imbalance_ratio"] = max(1.05, min(3.0, float(t["miss_imbalance_ratio"])))
    t["max_history_lines"] = max(50, min(5000, int(t["max_history_lines"])))
    return t


def load_auto_calibration_tuning(out_path: Path) -> Dict[str, Any]:
    """从已有 calibration_overrides.json 读取 auto_calibration 并与默认合并。"""
    base = dict(DEFAULT_AUTO_CALIBRATION)
    if out_path.is_file():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            ov = data.get("auto_calibration")
            if isinstance(ov, dict):
                for k, v in ov.items():
                    if k in base and v is not None:
                        if isinstance(base[k], int):
                            base[k] = int(v)
                        else:
                            base[k] = float(v)
        except (OSError, ValueError, TypeError, KeyError):
            pass
    return _sanitize_auto_tuning(base)


def _load_reconcile_history(data_dir: Path, max_lines: int) -> List[Dict[str, Any]]:
    path = data_dir / "reconcile_history.jsonl"
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


def _by_trade_date(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_d: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        d = rec.get("trade_date")
        if isinstance(d, str) and d:
            by_d[d] = rec
    return by_d


DEFAULT_THRESHOLDS = {
    "strong_buy": 6.5,
    "buy": 5.8,
    "hold": 5.0,
    "sell": 4.3,
}
DEFAULT_WEIGHTS = {
    "technical": 0.40,
    "fundamental": 0.35,
    "sentiment": 0.15,
    "sector": 0.10,
}


def _enforce_order(th: Dict[str, float]) -> Dict[str, float]:
    """保证 strong_buy > buy > hold > sell，且各落在合理区间。
    档位间距放宽到 +0.5（原 +1.0 过宽，会把收窄后的持有带弹回，抑制方向性信号）。"""
    s = max(2.5, min(4.5, float(th.get("sell", 3.5))))
    h = max(s + 0.5, min(6.2, float(th.get("hold", 5.0))))
    b = max(h + 0.5, min(8.2, float(th.get("buy", 5.8))))
    sb = max(b + 0.5, min(9.5, float(th.get("strong_buy", 6.5))))
    return {
        "sell": round(s, 3),
        "hold": round(h, 3),
        "buy": round(b, 3),
        "strong_buy": round(sb, 3),
    }


def _weighted_miss_rate(miss_w: float, w_sum: float) -> float:
    if w_sum <= 0:
        return 0.0
    return miss_w / w_sum


def _clamp_to_defaults(
    th: Dict[str, float],
    w: Dict[str, float],
    tuning: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    max_th = float(tuning["max_thresh_drift"])
    max_w = float(tuning["max_weight_drift"])
    th2 = {}
    for k, dv in DEFAULT_THRESHOLDS.items():
        v = th.get(k, dv)
        th2[k] = max(dv - max_th, min(dv + max_th, v))
    w2 = {}
    for k, dv in DEFAULT_WEIGHTS.items():
        v = w.get(k, dv)
        w2[k] = max(dv - max_w, min(dv + max_w, v))
    s = sum(w2.values())
    if s <= 0:
        w2 = dict(DEFAULT_WEIGHTS)
    else:
        w2 = {k: round(w2[k] / s, 4) for k in w2}
    return _enforce_order(th2), w2


def _bucket_ready(n_raw: int, w_sum: float, tuning: Dict[str, Any]) -> bool:
    return n_raw >= int(tuning["min_bucket_samples"]) or w_sum >= float(
        tuning["min_bucket_weight"]
    )


def _aggregate_stats(records: List[Dict[str, Any]], tuning: Dict[str, Any]) -> Dict[str, Any]:
    decay = float(tuning["decay_per_session"])
    last_n = int(tuning["last_sessions"])
    by_date = _by_trade_date(records)
    dates = sorted(by_date.keys())[-last_n:]
    n_sess = len(dates)
    stats: Dict[str, Any] = {
        "n_buy": 0,
        "miss_buy": 0,
        "n_sell": 0,
        "miss_sell": 0,
        "n_hold": 0,
        "miss_hold": 0,
        "n_strong_buy": 0,
        "miss_strong_buy": 0,
        "n_norm_buy": 0,
        "miss_norm_buy": 0,
        "n_strong_sell": 0,
        "miss_strong_sell": 0,
        "n_norm_sell": 0,
        "miss_norm_sell": 0,
        "w_buy": 0.0,
        "miss_buy_w": 0.0,
        "w_sell": 0.0,
        "miss_sell_w": 0.0,
        "w_hold": 0.0,
        "miss_hold_w": 0.0,
        "w_strong_buy": 0.0,
        "miss_strong_buy_w": 0.0,
        "w_norm_buy": 0.0,
        "miss_norm_buy_w": 0.0,
        "w_strong_sell": 0.0,
        "miss_strong_sell_w": 0.0,
        "w_norm_sell": 0.0,
        "miss_norm_sell_w": 0.0,
        "sessions_used": n_sess,
        "decay_per_session": decay,
        "last_sessions_cap": last_n,
    }
    for i, d in enumerate(dates):
        w_sess = float(decay) ** (n_sess - 1 - i) if n_sess else 1.0
        rec = by_date[d]
        for sym, info in (rec.get("symbols") or {}).items():
            if not isinstance(info, dict) or not info.get("ok", True):
                continue
            if info.get("match") is None:
                continue
            sig = str(info.get("signal") or "")
            exp = _expected_direction(sig)
            ok = bool(info["match"])
            if exp == 1:
                stats["n_buy"] += 1
                stats["w_buy"] += w_sess
                if not ok:
                    stats["miss_buy"] += 1
                    stats["miss_buy_w"] += w_sess
                if "强烈买入" in sig:
                    stats["n_strong_buy"] += 1
                    stats["w_strong_buy"] += w_sess
                    if not ok:
                        stats["miss_strong_buy"] += 1
                        stats["miss_strong_buy_w"] += w_sess
                elif sig == "买入":
                    stats["n_norm_buy"] += 1
                    stats["w_norm_buy"] += w_sess
                    if not ok:
                        stats["miss_norm_buy"] += 1
                        stats["miss_norm_buy_w"] += w_sess
            elif exp == -1:
                stats["n_sell"] += 1
                stats["w_sell"] += w_sess
                if not ok:
                    stats["miss_sell"] += 1
                    stats["miss_sell_w"] += w_sess
                if "强烈卖出" in sig:
                    stats["n_strong_sell"] += 1
                    stats["w_strong_sell"] += w_sess
                    if not ok:
                        stats["miss_strong_sell"] += 1
                        stats["miss_strong_sell_w"] += w_sess
                elif sig == "卖出":
                    stats["n_norm_sell"] += 1
                    stats["w_norm_sell"] += w_sess
                    if not ok:
                        stats["miss_norm_sell"] += 1
                        stats["miss_norm_sell_w"] += w_sess
            else:
                stats["n_hold"] += 1
                stats["w_hold"] += w_sess
                if not ok:
                    stats["miss_hold"] += 1
                    stats["miss_hold_w"] += w_sess
    return stats


def run_auto_calibration(stock_system_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    读取 data/reconcile_history.jsonl，更新 config/calibration_overrides.json。
    样本不足时仍写出 meta，阈值/权重保持与默认一致。
    """
    root = stock_system_root or Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    cfg_dir = root / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg_dir / "calibration_overrides.json"

    tuning = load_auto_calibration_tuning(out_path)
    records = _load_reconcile_history(
        data_dir, int(tuning["max_history_lines"])
    )
    stats = _aggregate_stats(records, tuning)
    # 校准门槛改用「方向性样本」（买入+卖出），持有信号不计入，
    # 避免「持有天然判对」注水阻止真实校准触发。
    directional_total = stats["n_buy"] + stats["n_sell"]
    min_total = int(tuning["min_total_samples"])
    step_th = float(tuning["step_th"])
    step_w = float(tuning["step_w"])
    hi = float(tuning["miss_rate_high"])
    lo = float(tuning["miss_rate_low"])
    imb = float(tuning["miss_imbalance_ratio"])

    base_th = dict(DEFAULT_THRESHOLDS)
    base_w = dict(DEFAULT_WEIGHTS)
    th, w = deepcopy(base_th), deepcopy(base_w)
    notes: List[str] = []

    if directional_total < min_total:
        notes.append(
            f"方向性样本不足（买入+卖出有效样本 {directional_total} < {min_total}），未调整阈值/权重。"
        )
    else:
        touched_buy = False
        if _bucket_ready(
            int(stats["n_strong_buy"]), float(stats["w_strong_buy"]), tuning
        ):
            mr = _weighted_miss_rate(
                float(stats["miss_strong_buy_w"]), float(stats["w_strong_buy"])
            )
            if mr > hi:
                th["strong_buy"] += step_th
                notes.append(f"强烈买入加权误判率偏高({mr:.0%})，略提高 strong_buy 门槛。")
                touched_buy = True
            elif mr < lo:
                th["strong_buy"] -= step_th
                notes.append(f"强烈买入加权较准({mr:.0%})，略放宽 strong_buy。")
                touched_buy = True

        if _bucket_ready(int(stats["n_norm_buy"]), float(stats["w_norm_buy"]), tuning):
            mr = _weighted_miss_rate(
                float(stats["miss_norm_buy_w"]), float(stats["w_norm_buy"])
            )
            if mr > hi:
                th["buy"] += step_th
                notes.append(f"普通买入加权误判率偏高({mr:.0%})，略提高 buy 门槛。")
                touched_buy = True
            elif mr < lo:
                th["buy"] -= step_th
                notes.append(f"普通买入加权较准({mr:.0%})，略放宽 buy。")
                touched_buy = True

        if not touched_buy and _bucket_ready(
            int(stats["n_buy"]), float(stats["w_buy"]), tuning
        ):
            mr = _weighted_miss_rate(float(stats["miss_buy_w"]), float(stats["w_buy"]))
            if mr > hi:
                th["buy"] += step_th
                th["strong_buy"] += step_th
                notes.append(f"买入向加权误判率偏高({mr:.0%})，略提高买入门槛。")
            elif mr < lo:
                th["buy"] -= step_th
                th["strong_buy"] -= step_th
                notes.append(f"买入向加权较准({mr:.0%})，略放宽买入门槛。")

        touched_sell = False
        if _bucket_ready(
            int(stats["n_strong_sell"]), float(stats["w_strong_sell"]), tuning
        ):
            mr = _weighted_miss_rate(
                float(stats["miss_strong_sell_w"]), float(stats["w_strong_sell"])
            )
            if mr > hi:
                th["sell"] += step_th
                notes.append(
                    f"强烈卖出加权误判率偏高({mr:.0%})，略抬高 sell 线、收缩极端看空区。"
                )
                touched_sell = True
            elif mr < lo:
                th["sell"] -= step_th
                notes.append(f"强烈卖出加权较准({mr:.0%})，略放宽极端看空区。")
                touched_sell = True

        if _bucket_ready(int(stats["n_norm_sell"]), float(stats["w_norm_sell"]), tuning):
            mr = _weighted_miss_rate(
                float(stats["miss_norm_sell_w"]), float(stats["w_norm_sell"])
            )
            if mr > hi:
                th["hold"] -= step_th
                th["sell"] -= step_th
                notes.append(
                    f"普通卖出加权误判率偏高({mr:.0%})，略扩大持有带、减少边缘卖出。"
                )
                touched_sell = True
            elif mr < lo:
                th["hold"] += step_th
                th["sell"] += step_th
                notes.append(f"普通卖出加权较准({mr:.0%})，略收紧持有带。")
                touched_sell = True

        if not touched_sell and _bucket_ready(
            int(stats["n_sell"]), float(stats["w_sell"]), tuning
        ):
            mr = _weighted_miss_rate(float(stats["miss_sell_w"]), float(stats["w_sell"]))
            if mr > hi:
                th["hold"] -= step_th
                th["sell"] -= step_th
                notes.append(
                    f"卖出向加权误判率偏高({mr:.0%})，略扩大持有带、减少边缘卖出。"
                )
            elif mr < lo:
                th["hold"] += step_th
                th["sell"] += step_th
                notes.append(f"卖出向加权较准({mr:.0%})，略收紧持有带。")

        mbw = float(stats["miss_buy_w"])
        msw = float(stats["miss_sell_w"])
        if mbw + msw > 0:
            if mbw > msw * imb:
                w["technical"] -= step_w
                w["fundamental"] += step_w
                notes.append("买入误判（加权）更多，略降技术面、增基本面权重。")
            elif msw > mbw * imb:
                w["technical"] -= step_w
                w["sector"] += step_w
                notes.append("卖出误判（加权）更多，略降技术面、增行业权重。")

        if _bucket_ready(int(stats["n_hold"]), float(stats["w_hold"]), tuning):
            mr = _weighted_miss_rate(float(stats["miss_hold_w"]), float(stats["w_hold"]))
            if mr > hi:
                th["hold"] += step_th
                th["buy"] -= step_th
                notes.append(f"持有向加权误判率偏高({mr:.0%})，收窄评分意义上的持有带。")
            elif mr < lo:
                th["hold"] -= step_th
                th["buy"] += step_th
                notes.append(f"持有向加权较准({mr:.0%})，略放宽持有带。")

        th, w = _clamp_to_defaults(th, w, tuning)
        th = _enforce_order(th)

    payload: Dict[str, Any] = {
        "version": 1,
        "updated_at": datetime.now().isoformat(),
        "source": "reconcile_history.jsonl",
        "stats": stats,
        "notes": notes,
        "signal_thresholds": th,
        "score_weights": w,
        "auto_calibration": tuning,
    }
    if out_path.is_file():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                old = json.load(f)
            at = old.get("accuracy_tuning")
            if isinstance(at, dict) and at:
                payload["accuracy_tuning"] = at
        except (OSError, ValueError, TypeError):
            pass
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


if __name__ == "__main__":
    import sys

    r = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    p = run_auto_calibration(r)
    keys = (
        "updated_at",
        "stats",
        "notes",
        "signal_thresholds",
        "score_weights",
        "auto_calibration",
    )
    print(json.dumps({k: p[k] for k in keys}, ensure_ascii=False, indent=2))
