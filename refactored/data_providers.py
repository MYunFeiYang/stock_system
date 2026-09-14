#!/usr/bin/env python3
"""
股票分析数据源：仅通过 OpenClaw Agent（网页搜索/浏览工具）拉取公开行情摘要。

环境变量（可选）：
  OPENCLAW_BIN                 openclaw 可执行文件（默认 PATH）
  OPENCLAW_STOCK_AGENT_ID      默认 main
  OPENCLAW_AGENT_LOCAL         1=--local（默认）；0=走 Gateway
  OPENCLAW_AGENT_TIMEOUT       秒，默认 600
  STOCK_OPENCLAW_CACHE_SEC     单股缓存秒数，默认 90
  STOCK_OPENCLAW_MAX_ATTEMPTS  单股拉价最多调用 Agent 次数，默认 2
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Protocol

ANALYSIS_TYPE_NAMES = {
    "morning": "早盘",
    "afternoon": "午盘",
    "evening": "收盘",
    "weekly": "周度",
    "reconcile": "收盘复盘（对照早盘预测）",
    "day_review": "全日预测汇总",
    "post_close": "收盘复盘 + 全日汇总（含自校准）",
}


def get_analysis_type_name(analysis_type: str) -> str:
    """获取分析类型中文名称（统一入口）。"""
    return ANALYSIS_TYPE_NAMES.get(analysis_type, analysis_type)


SECTOR_BENCHMARKS = {
    "白酒": {"pe_avg": 28, "pb_avg": 7, "roe_avg": 22, "growth_avg": 15},
    "新能源": {"pe_avg": 35, "pb_avg": 4, "roe_avg": 16, "growth_avg": 25},
    "银行": {"pe_avg": 6, "pb_avg": 0.9, "roe_avg": 12, "growth_avg": 8},
    "医药": {"pe_avg": 32, "pb_avg": 4, "roe_avg": 15, "growth_avg": 18},
    "科技": {"pe_avg": 38, "pb_avg": 5, "roe_avg": 18, "growth_avg": 20},
    "消费": {"pe_avg": 26, "pb_avg": 4, "roe_avg": 16, "growth_avg": 12},
    "地产": {"pe_avg": 8, "pb_avg": 1.2, "roe_avg": 10, "growth_avg": 5},
    "面板": {"pe_avg": 15, "pb_avg": 2, "roe_avg": 12, "growth_avg": 10},
}


def _sector_bm(sector: str) -> Dict[str, float]:
    return SECTOR_BENCHMARKS.get(
        sector, {"pe_avg": 20, "pb_avg": 3, "roe_avg": 15, "growth_avg": 15}
    )


@dataclass
class StockInputs:
    current_price: float
    change_percent: float
    technical: Dict
    fundamental: Dict
    sentiment: Dict
    sector: Dict
    provenance: str


def _neutral_technical() -> Dict:
    """当 K 线数据不可用时的中性技术指标 fallback。"""
    return {
        "rsi": 50.0,
        "macd_signal": "中性",
        "bollinger_position": 0.5,
        "volume_ratio": 1.0,
        "momentum_5d": 0.0,
        "momentum_10d": 0.0,
        "momentum_20d": 0.0,
        "ma_alignment": "中性",
        "ma5": None,
        "ma10": None,
        "ma20": None,
        "ma60": None,
    }


# ── 真实技术指标计算（基于 K 线历史数据） ──────────────────────────


def _ema(values: list, period: int) -> list:
    """计算指数移动平均"""
    if len(values) < period:
        return [sum(values) / len(values)] * len(values)
    result = [sum(values[:period]) / period]
    multiplier = 2.0 / (period + 1)
    for v in values[period:]:
        result.append(v * multiplier + result[-1] * (1 - multiplier))
    return [result[0]] * (period - 1) + result


def _sma(values: list, period: int) -> list:
    """简单移动平均"""
    if len(values) < period:
        return [sum(values) / len(values)] * len(values)
    result = []
    for i in range(len(values)):
        if i < period - 1:
            result.append(sum(values[: i + 1]) / (i + 1))
        else:
            result.append(sum(values[i - period + 1 : i + 1]) / period)
    return result


def compute_rsi(closes: list, period: int = 14) -> float:
    """计算 RSI（Relative Strength Index）"""
    if len(closes) < period + 1:
        return 50.0
    gains = []
    losses = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(diff if diff > 0 else 0.0)
        losses.append(-diff if diff < 0 else 0.0)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - 100.0 / (1.0 + rs), 1)


def compute_macd(closes: list, fast: int = 12, slow: int = 26, signal: int = 9):
    """返回 (DIF, DEA, MACD_bar, signal_text)"""
    if len(closes) < slow + signal:
        return 0.0, 0.0, 0.0, "中性"
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    dif = [f - s for f, s in zip(ema_fast, ema_slow)]
    dea = _ema(dif, signal)
    bar = 2.0 * (dif[-1] - dea[-1])

    # 判断信号
    if dif[-1] > dea[-1] and dif[-2] <= dea[-2]:
        sig = "金叉"
    elif dif[-1] < dea[-1] and dif[-2] >= dea[-2]:
        sig = "死叉"
    elif dif[-1] > dea[-1]:
        sig = "多头"
    else:
        sig = "空头"
    return round(dif[-1], 4), round(dea[-1], 4), round(bar, 4), sig


def compute_bollinger(closes: list, period: int = 20, std_mult: float = 2.0):
    """返回 (upper, middle, lower, position)，position 0=下轨 1=上轨"""
    if len(closes) < period:
        return None, None, None, 0.5
    ma = _sma(closes, period)
    middle = ma[-1]
    # 计算标准差
    recent = closes[-period:]
    mean = sum(recent) / period
    variance = sum((x - mean) ** 2 for x in recent) / period
    std = variance ** 0.5
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    pos = (closes[-1] - lower) / (upper - lower) if upper != lower else 0.5
    return round(upper, 2), round(middle, 2), round(lower, 2), round(max(0.0, min(1.0, pos)), 3)


def compute_volume_ratio(volumes: list, period: int = 5) -> float:
    """量比 = 今日成交量 / N日均量"""
    if len(volumes) < period:
        return 1.0
    avg_vol = sum(volumes[-(period + 1) : -1]) / period
    if avg_vol == 0:
        return 1.0
    return round(volumes[-1] / avg_vol, 2)


def compute_ma_alignment(prices: list, periods: tuple = (5, 10, 20, 60)):
    """判断均线排列：多头排列 / 空头排列 / 缠绕（中性）"""
    mas = {}
    for p in periods:
        if len(prices) >= p:
            mas[f"ma{p}"] = round(sum(prices[-p:]) / p, 2)
        else:
            mas[f"ma{p}"] = None

    valid_mas = [mas[f"ma{p}"] for p in periods if mas[f"ma{p}"] is not None]
    if len(valid_mas) >= 3:
        # 多头排列：短期 > 中期 > 长期
        if all(valid_mas[i] > valid_mas[i + 1] for i in range(len(valid_mas) - 1)):
            alignment = "多头排列"
        elif all(valid_mas[i] < valid_mas[i + 1] for i in range(len(valid_mas) - 1)):
            alignment = "空头排列"
        else:
            alignment = "均线缠绕"
    else:
        alignment = "中性"

    return mas, alignment


def compute_real_technicals(klines: list) -> Dict:
    """
    基于真实 K 线数据计算全套技术指标。

    klines: list of dict, each has keys: open, close, high, low, volume (all strings)
    返回完整的 technical dict。
    """
    if not klines or len(klines) < 14:
        return _neutral_technical()

    closes = []
    volumes = []
    for k in klines:
        try:
            closes.append(float(k.get("close", 0)))
            volumes.append(float(k.get("volume", 0)))
        except (ValueError, TypeError):
            continue

    if len(closes) < 14:
        return _neutral_technical()

    # RSI (14)
    rsi = compute_rsi(closes)

    # MACD
    dif, dea, macd_bar, macd_signal = compute_macd(closes)

    # 布林带
    bb_upper, bb_mid, bb_lower, bb_pos = compute_bollinger(closes)

    # 量比
    vol_ratio = compute_volume_ratio(volumes)

    # 多周期动量
    def momentum(cl, days):
        if len(cl) > days and cl[-1 - days] > 0:
            return round((cl[-1] - cl[-1 - days]) / cl[-1 - days], 4)
        return 0.0

    mom5 = momentum(closes, 5)
    mom10 = momentum(closes, 10)
    mom20 = momentum(closes, 20)

    # 均线排列
    mas, alignment = compute_ma_alignment(closes)

    return {
        "rsi": rsi,
        "macd_signal": macd_signal,
        "macd_dif": dif,
        "macd_dea": dea,
        "macd_bar": macd_bar,
        "bollinger_position": bb_pos,
        "bollinger_upper": bb_upper,
        "bollinger_mid": bb_mid,
        "bollinger_lower": bb_lower,
        "volume_ratio": vol_ratio,
        "momentum_5d": mom5,
        "momentum_10d": mom10,
        "momentum_20d": mom20,
        "ma_alignment": alignment,
        "ma5": mas.get("ma5"),
        "ma10": mas.get("ma10"),
        "ma20": mas.get("ma20"),
        "ma60": mas.get("ma60"),
    }


def _std(values: list) -> float:
    """样本标准差。"""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(var)


def detect_market_status(klines: list) -> Dict:
    """
    基于宽基指数 K 线判定市场状态（regime）。

    返回 dict：
      - status: trending_up | trending_down | ranging | volatile_up | volatile_down
      - regime: trend | range | volatility （粗分三类，供权重调整用）
      - volatility: 年化波动率估计（0.2 = 20%）
      - trend_strength: 20日动量方向，约 -0.2..0.2
      - bollinger_bandwidth: 布林带宽占中轨比，越小越震荡
      - description: 中文说明

    判定逻辑：
      - 年化波动率 > 35% → volatility（高波动，风控优先）
      - |20日动量| >= 4% 且 布林带宽 >= 4% → trend（有方向）
      - 否则 → range（震荡，均值回归）
    """
    default = {
        "status": "ranging",
        "regime": "range",
        "volatility": 0.0,
        "trend_strength": 0.0,
        "bollinger_bandwidth": 0.0,
        "description": "数据不足，默认震荡市",
    }
    if not klines or len(klines) < 20:
        return default

    closes = []
    for k in klines:
        try:
            closes.append(float(k.get("close", 0)))
        except (ValueError, TypeError):
            continue
    if len(closes) < 20:
        return default

    # 年化波动率（日收益标准差 × √242 交易日）
    rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
    vol_daily = _std(rets[-20:])
    vol_annual = vol_daily * math.sqrt(242)

    # 20 日动量（趋势方向与强度）
    if len(closes) > 21 and closes[-22] > 0:
        mom20 = (closes[-1] - closes[-22]) / closes[-22]
    else:
        mom20 = (closes[-1] - closes[-2]) / closes[-2] if closes[-2] else 0.0

    # 布林带宽
    bb_upper, bb_mid, bb_lower, _ = compute_bollinger(closes)
    bbw = (bb_upper - bb_lower) / bb_mid if bb_mid else 0.0

    # 判定 regime
    if vol_annual > 0.35:
        regime = "volatility"
    elif abs(mom20) >= 0.04 and bbw >= 0.04:
        regime = "trend"
    else:
        regime = "range"

    if regime == "trend":
        status = "trending_up" if mom20 > 0 else "trending_down"
    elif regime == "volatility":
        status = "volatile_up" if mom20 > 0 else "volatile_down"
    else:
        status = "ranging"

    desc = {
        "trending_up": "趋势上涨市：顺势而为，技术面占优",
        "trending_down": "趋势下跌市：顺势防御，技术面占优",
        "ranging": "震荡市：均值回归，基本面/估值占优",
        "volatile_up": "高波动偏多：风控优先，关注情绪面",
        "volatile_down": "高波动偏空：风控优先，关注情绪面",
    }.get(status, status)

    return {
        "status": status,
        "regime": regime,
        "volatility": round(vol_annual, 3),
        "trend_strength": round(mom20, 4),
        "bollinger_bandwidth": round(bbw, 4),
        "description": desc,
    }


def technical_from_spot_change(change_percent: float) -> Dict:
    """由涨跌幅构造技术面代理指标（fallback，当 K 线不可用时）。"""
    t = _neutral_technical()
    t["momentum_5d"] = round(max(-0.05, min(0.05, change_percent / 500.0)), 4)
    if change_percent > 2:
        t["macd_signal"] = "金叉"
    elif change_percent < -2:
        t["macd_signal"] = "死叉"
    return t


def _neutral_fundamental(stock: Any) -> Dict:
    b = _sector_bm(getattr(stock, "sector", "") or "")
    return {
        "pe_ratio": float(b["pe_avg"]),
        "pb_ratio": float(b["pb_avg"]),
        "roe": float(b["roe_avg"]),
        "growth_rate": float(b["growth_avg"]),
        "debt_ratio": 0.45,
        "dividend_yield": 2.0,
    }


def sentiment_from_price_action(change_percent: float, turnover: float) -> Dict:
    """由涨跌幅与换手率构造情绪代理指标（无独立舆情 API 时）。"""
    heat = int(min(10, max(1, round(5 + abs(change_percent) / 3))))
    att = int(min(10, max(1, round(3 + min(turnover, 15) / 2))))
    if change_percent > 2:
        retail = "乐观"
    elif change_percent < -2:
        retail = "谨慎"
    else:
        retail = "中性"
    if change_percent > 1:
        news = "正面"
    elif change_percent < -1:
        news = "负面"
    else:
        news = "中性"
    return {
        "market_heat": heat,
        "institution_attention": att,
        "retail_sentiment": retail,
        "news_sentiment": news,
    }


def sector_from_price_action(change_percent: float) -> Dict:
    """由涨跌幅构造行业轮动代理维度（fallback，当相对强度不可用时）。"""
    base = 5.0 + min(3.0, max(-3.0, change_percent / 2))
    p = int(min(9, max(3, round(base + 1))))
    return {
        "prosperity": p,
        "policy_support": int(min(9, max(3, round(6 + change_percent / 4)))),
        "capital_flow": int(min(9, max(2, round(5 + change_percent / 3)))),
        "rotation_position": int(min(9, max(3, round(5 + abs(change_percent) / 5)))),
    }


def _lin(x: float, points: list) -> float:
    """分段线性插值。points: [(x0,y0),(x1,y1),...]（按 x 升序）。"""
    if not points:
        return 0.0
    if x <= points[0][0]:
        return float(points[0][1])
    if x >= points[-1][0]:
        return float(points[-1][1])
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
            return y0 + t * (y1 - y0)
    return float(points[-1][1])


def sentiment_from_technical(technical: Dict) -> Dict:
    """由真实技术指标构造情绪面（放量/缩量 + 均线排列 + 中期动量）。

    相比 sentiment_from_price_action 直接用单日涨跌幅「造假」，这里用
    K 线衍生的 volume_ratio / ma_alignment / momentum_20d（均为真实信号）。
    返回字段与 calculate_sentiment_score 期望一致。
    """
    vr = float(technical.get("volume_ratio", 1.0))
    align = str(technical.get("ma_alignment", "中性"))
    mom20 = float(technical.get("momentum_20d", 0))

    # 放量上涨偏乐观，放量下跌偏恐慌，缩量观望
    if vr > 1.5 and mom20 > 0:
        base = 7.0
    elif vr > 1.5 and mom20 < 0:
        base = 3.5
    elif vr < 0.7:
        base = 4.5
    else:
        base = 5.0

    # 均线排列修正
    if align == "多头排列":
        base = min(9.0, base + 1.0)
    elif align == "空头排列":
        base = max(2.0, base - 1.0)

    base = round(base, 1)
    return {
        "market_heat": base,
        "institution_attention": base,
        "retail_sentiment": "乐观" if base >= 7 else "谨慎" if base <= 4 else "中性",
        "news_sentiment": "正面" if base >= 7 else "负面" if base <= 4 else "中性",
    }


def relative_strength_score(stock_mom20: float, index_mom20: float) -> float:
    """个股 20 日动量相对宽基指数的强弱（真实相对强度 / RS 信号）。

    返回 1-10：个股显著跑赢指数为高，跑输为低。
    指数动量来自 detect_market_status（P2，真实 K 线）。
    """
    rs = float(stock_mom20) - float(index_mom20)
    return round(_lin(rs, [(-0.10, 2), (-0.03, 4), (0, 5), (0.03, 6.5), (0.10, 8.5)]), 1)


class StockDataProvider(Protocol):
    def fetch(self, stock: Any) -> StockInputs: ...


def get_default_provider() -> StockDataProvider:
    """返回默认数据提供者。

    生产环境（STOCK_SKIP_AGENT=1）直接使用 akshare/sina HTTP fallback，
    不启动 OpenClaw Agent 以节省时间和资源。
    """
    skip_agent = os.environ.get("STOCK_SKIP_AGENT", "").lower() in ("1", "true", "yes")
    if skip_agent:
        from akshare_fallback import get_akshare_provider
        return get_akshare_provider()

    from openclaw_search_provider import OpenclawAgentWebProvider
    return OpenclawAgentWebProvider()
