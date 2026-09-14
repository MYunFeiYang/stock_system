#!/usr/bin/env python3
"""
行情直取 fallback —— 当 OpenClaw Agent 不可用时，直接通过 HTTP 调用行情接口
拉取 A 股现价与涨跌幅。

三层策略（逐源补齐，单源故障不致全废）：
  1. akshare（可选，安装后自动优先）
  2. 新浪财经 HTTP（零依赖，py3 内置 urllib）—— 批量单次可查多只
  3. 腾讯财经 HTTP（零依赖，同 urllib）—— 新浪不可达时兜底

环境变量：
  STOCK_FALLBACK_CACHE_SEC  单股缓存秒数，默认 60
"""

from __future__ import annotations

import os
import re
import time
import urllib.request
from typing import Any, Dict, Optional, Tuple

from data_providers import (
    StockInputs,
    _neutral_fundamental,
    compute_real_technicals,
    sector_from_price_action,
    sentiment_from_technical,
    technical_from_spot_change,
)


# ── Layer 1: 新浪财经 HTTP 直连（零依赖） ──────────────────────────────

_SINA_QUOTE_URL = "https://hq.sinajs.cn/list={codes}"
_SINA_KLINE_URL = (
    "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "CN_MarketData.getKLineData?symbol={symbol}&scale=240&ma=no&datalen={days}"
)


def _fetch_sina_kline(symbol: str, days: int = 60) -> list:
    """拉取新浪日线 K 线，返回 list of dict（含 open/close/high/low/volume/amount/day）。"""
    url = _SINA_KLINE_URL.format(symbol=symbol, days=days)
    try:
        req = urllib.request.Request(url, headers={"Referer": "https://finance.sina.com.cn"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("gbk", errors="replace")
        import json

        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []


def fetch_index_klines(symbol: str, days: int = 60) -> list:
    """
    拉取宽基指数日线 K 线，用于市场状态判定。

    symbol 用新浪格式（如 sh000001 上证指数、sz399006 创业板指、
    sh000300 沪深300）。优先 akshare，降级新浪 HTTP（零依赖）。
    """
    if _akshare_available():
        try:
            import akshare as ak
            df = ak.index_zh_a_hist(
                symbol=symbol.lstrip("shsz"),
                period="daily",
                start_date="20240101",
                end_date="20261231",
            )
            if df is not None and not df.empty:
                out = []
                for _, row in df.iterrows():
                    out.append({
                        "day": str(row.get("日期", "")),
                        "open": str(row.get("开盘", 0)),
                        "close": str(row.get("收盘", 0)),
                        "high": str(row.get("最高", 0)),
                        "low": str(row.get("最低", 0)),
                        "volume": str(row.get("成交量", 0)),
                    })
                return out[-days:]
        except Exception:
            pass
    # 新浪指数 K 线：代码需带交易所前缀
    return _fetch_sina_kline(symbol, days)


def _code_to_sina_symbol(code: str) -> str:
    """600519 → sh600519, 000858 → sz000858"""
    code = code.strip().zfill(6)
    if code.startswith(("6", "9")):
        return f"sh{code}"
    elif code.startswith(("0", "3")):
        return f"sz{code}"
    else:
        return f"sh{code}"  # default to SH


def _parse_sina_line(line: str) -> Optional[Tuple[str, float, float]]:
    """
    解析新浪行情行，返回 (代码, 现价, 涨跌幅%)。
    格式: var hq_str_sh600519="name,open,prev_close,price,..."
    """
    m = re.match(r'var hq_str_(\w+)="(.+)"', line)
    if not m:
        return None
    symbol = m.group(1)  # e.g., "sh600519"
    fields = m.group(2).split(",")
    if len(fields) < 4:
        return None

    code = symbol[2:]  # strip "sh"/"sz" prefix
    try:
        prev_close = float(fields[2])  # 昨收
        price = float(fields[3])       # 现价
    except (ValueError, TypeError):
        return None

    if price <= 0 or prev_close <= 0:
        return None

    pct = round((price - prev_close) / prev_close * 100, 2)
    return (code, price, pct)


def _fetch_sina_batch(codes: list) -> Dict[str, Tuple[float, float]]:
    """批量拉取新浪行情，返回 {代码: (现价, 涨跌幅%)}。"""
    if not codes:
        return {}

    symbols = ",".join(_code_to_sina_symbol(c) for c in codes)
    url = _SINA_QUOTE_URL.format(codes=symbols)

    try:
        req = urllib.request.Request(
            url,
            headers={"Referer": "https://finance.sina.com.cn"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("gbk", errors="replace")
    except Exception:
        return {}

    result: Dict[str, Tuple[float, float]] = {}
    for line in raw.strip().split("\n"):
        parsed = _parse_sina_line(line.strip())
        if parsed:
            code, price, pct = parsed
            result[code] = (price, pct)
    return result


def _fetch_tencent_batch(codes: list) -> Dict[str, Tuple[float, float]]:
    """腾讯财经 HTTP 直连兜底（零依赖）。新浪不可达时补位。

    接口 qt.gtimg.cn：v_sh600519="1~名称~代码~现价~昨收~今开~..."（~ 分隔）。
    涨跌幅按 (现价-昨收)/昨收 计算，与新浪口径一致。
    """
    if not codes:
        return {}
    syms = ",".join(
        ("sh" if c.strip().zfill(6)[0] in "69" else "sz") + c.strip().zfill(6)
        for c in codes
    )
    url = f"https://qt.gtimg.cn/q={syms}"
    try:
        req = urllib.request.Request(url, headers={"Referer": "https://finance.qq.com"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("gbk", errors="replace")
    except Exception:
        return {}

    result: Dict[str, Tuple[float, float]] = {}
    for line in raw.strip().split("\n"):
        m = re.match(r"v_(\w+)=\"(.+)\"", line)
        if not m:
            continue
        sym = m.group(1)          # e.g. "sh600519"
        fields = m.group(2).split("~")
        if len(fields) < 5:
            continue
        code = sym[2:]
        try:
            price = float(fields[3])     # 现价
            prev_close = float(fields[4])  # 昨收
        except (ValueError, TypeError):
            continue
        if price > 0 and prev_close > 0:
            pct = round((price - prev_close) / prev_close * 100, 2)
            result[code] = (price, pct)
    return result


def _fetch_quote_batch(codes: list) -> Dict[str, Tuple[float, float]]:
    """统一行情抓取：优先 akshare → 新浪 → 腾讯，逐源补齐缺失代码。

    任一单源故障（超时/被封/空响应）都不会导致整批失效——后续源自动兜底。
    返回 {代码: (现价, 涨跌幅%)}。
    """
    if not codes:
        return {}
    data: Dict[str, Tuple[float, float]] = {}

    def _missing() -> list:
        return [c for c in codes if c not in data]

    if _akshare_available():
        try:
            data.update(_fetch_akshare_batch(codes))
        except Exception:
            pass
    if _missing():
        try:
            data.update(_fetch_sina_batch(_missing()))
        except Exception:
            pass
    if _missing():
        try:
            data.update(_fetch_tencent_batch(_missing()))
        except Exception:
            pass
    return data


# ── Layer 2: akshare（可选增强）───────────────────────────────────────

def _akshare_available() -> bool:
    try:
        import akshare  # noqa: F401
        return True
    except ImportError:
        return False


def _fetch_akshare_batch(codes: list) -> Dict[str, Tuple[float, float]]:
    """通过 akshare 拉取批量行情。"""
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        if df is None or df.empty:
            return {}
        result: Dict[str, Tuple[float, float]] = {}
        code_set = set(c.strip().zfill(6) for c in codes)
        for _, row in df.iterrows():
            code = str(row.get("代码", "")).strip().zfill(6)
            if code not in code_set:
                continue
            price = float(row.get("最新价", 0) or 0)
            pct = float(row.get("涨跌幅", 0) or 0)
            if code and price > 0:
                result[code] = (price, pct)
        return result
    except Exception:
        return {}


def _fetch_akshare_kline(code: str, days: int = 60) -> list:
    """通过 akshare 拉取日线 K 线（优先使用，更准）。"""
    try:
        import akshare as ak
        df = ak.stock_zh_a_hist(
            symbol=code, period="daily", adjust="qfq",
            start_date="20240101", end_date="20261231",
        )
        if df is None or df.empty:
            return []
        klines = []
        # akshare 列名：日期,开盘,收盘,最高,最低,成交量,成交额
        for _, row in df.iterrows():
            klines.append({
                "day": str(row.get("日期", "")),
                "open": str(row.get("开盘", 0)),
                "close": str(row.get("收盘", 0)),
                "high": str(row.get("最高", 0)),
                "low": str(row.get("最低", 0)),
                "volume": str(row.get("成交量", 0)),
            })
        return klines[-days:]
    except Exception:
        return []


def fetch_daily_ohlc(symbol: str, ymd: str, retries: int | None = None) -> Optional[Tuple[float, float, float]]:
    """返回交易日 ymd 的 (开盘价, 收盘价, 昨收价)，用于收盘复盘计算真实区间涨跌（相对昨收口径）。

    与实时快照不同，交易日的开/收是收盘后即固定的历史值，任何时刻（含 cron 因
    睡眠延迟到收盘后运行）都能取到，因此复盘不再受运行时刻影响而恒为 0。

    优先 akshare stock_zh_a_hist（按日期过滤，失败再全量扫描兜底），
    降级新浪日线 K 线取当日；均失败返回 None。带指数退避重试。
    """
    code = symbol.strip().zfill(6)
    if retries is None:
        retries = int(os.environ.get("STOCK_FETCH_RETRIES") or "3")
    retries = max(1, retries)
    ymd = str(ymd).replace("-", "")

    # ── 优先 akshare ──
    if _akshare_available():
        # 1) 按日期精确过滤（最快）
        for attempt in range(retries):
            try:
                import akshare as ak
                df = ak.stock_zh_a_hist(
                    symbol=code, period="daily", adjust="qfq",
                    start_date=ymd, end_date=ymd,
                )
                if df is not None and not df.empty:
                    row = df.iloc[0]
                    o = float(row.get("开盘", 0) or 0)
                    c = float(row.get("收盘", 0) or 0)
                    if o > 0 and c > 0:
                        prev_close = c - float(row.get("涨跌额", 0) or 0)
                        return (o, c, prev_close)
            except Exception:
                pass
            if attempt < retries - 1:
                time.sleep(min(4.0, 0.5 * (2 ** attempt)))
        # 2) 全量扫描兜底（akshare 单日过滤偶发不稳）
        try:
            import akshare as ak
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily", adjust="qfq",
                start_date="20240101", end_date="20261231",
            )
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    if str(row.get("日期", "")).replace("-", "") == ymd:
                        o = float(row.get("开盘", 0) or 0)
                        c = float(row.get("收盘", 0) or 0)
                        if o > 0 and c > 0:
                            prev_close = c - float(row.get("涨跌额", 0) or 0)
                            return (o, c, prev_close)
        except Exception:
            pass

    # ── 降级新浪日线 K 线 ──
    sina_sym = _code_to_sina_symbol(code)
    for attempt in range(retries):
        for k in _fetch_sina_kline(sina_sym, 90):
            if str(k.get("day", "")).replace("-", "") == ymd:
                try:
                    o = float(k.get("open", 0) or 0)
                    c = float(k.get("close", 0) or 0)
                except (ValueError, TypeError):
                    o = c = 0
                if o > 0 and c > 0:
                    prev_close = float(k.get("preClose", 0) or 0)
                    return (o, c, prev_close)
        if attempt < retries - 1:
            time.sleep(min(4.0, 0.5 * (2 ** attempt)))
    return None


def fetch_recent_ohlc(symbol: str, n: int = 2, retries: int | None = None) -> Optional[list[Tuple[float, float]]]:
    """返回最近 n 个【已收盘】交易日的 [(开盘, 收盘), ...]（升序，末位为最近一个交易日）。

    用于早盘预测锚定"上一交易日收盘"：交易日的开/收是收盘后即固定的历史值，任何时刻
    （含 cron 因睡眠延迟到盘中才跑）取到都一样，预测因此与运行时刻解耦。
    自动跳过今天（可能未收盘，避免取到盘中临时值）。
    akshare 全量历史优先，降级新浪日线 K 线；不足 n 天或失败返回 None。带指数退避重试。
    """
    code = symbol.strip().zfill(6)
    if retries is None:
        retries = int(os.environ.get("STOCK_FETCH_RETRIES") or "3")
    retries = max(1, retries)
    today_ymd = time.strftime("%Y%m%d", time.localtime())

    # ── 优先 akshare 全量历史 ──
    if _akshare_available():
        for attempt in range(retries):
            try:
                import akshare as ak
                df = ak.stock_zh_a_hist(
                    symbol=code, period="daily", adjust="qfq",
                    start_date="20240101", end_date="20261231",
                )
                if df is not None and not df.empty:
                    pairs = []
                    for _, row in df.iterrows():
                        d = str(row.get("日期", "")).replace("-", "")
                        if d >= today_ymd:
                            continue
                        o = float(row.get("开盘", 0) or 0)
                        c = float(row.get("收盘", 0) or 0)
                        if o > 0 and c > 0:
                            pairs.append((o, c))
                    if len(pairs) >= n:
                        return pairs[-n:]
            except Exception:
                pass
            if attempt < retries - 1:
                time.sleep(min(4.0, 0.5 * (2 ** attempt)))

    # ── 降级新浪日线 K 线 ──
    sina_sym = _code_to_sina_symbol(code)
    for attempt in range(retries):
        kl = list(_fetch_sina_kline(sina_sym, 120))
        pairs = []
        for k in kl:
            d = str(k.get("day", "")).replace("-", "")[:8]
            if d >= today_ymd:
                continue
            try:
                o = float(k.get("open", 0) or 0)
                c = float(k.get("close", 0) or 0)
            except (ValueError, TypeError):
                o = c = 0
            if o > 0 and c > 0:
                pairs.append((o, c))
        if len(pairs) >= n:
            return pairs[-n:]
        if attempt < retries - 1:
            time.sleep(min(4.0, 0.5 * (2 ** attempt)))
    return None


# ── 统一接口 ─────────────────────────────────────────────────────────

class AkshareFallbackProvider:
    """
    行情 fallback 提供者。
    优先 akshare → 降级新浪财经 HTTP（零依赖）。

    不依赖 OpenClaw Agent / Gateway，独立运行。
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[float, Tuple[float, float]]] = {}
        self._ttl = float(os.environ.get("STOCK_FALLBACK_CACHE_SEC") or "60")

    def fetch_one(self, code: str, retries: int | None = None) -> Optional[Tuple[float, float]]:
        """返回 (现价, 涨跌幅%)，失败返回 None。

        对底层 HTTP/akshare 拉取做指数退避重试，减少偶发网络抖动导致整批中断。
        """
        code = code.strip().zfill(6)
        now = time.time()
        if code in self._cache:
            ts, pair = self._cache[code]
            if now - ts < self._ttl:
                return pair

        if retries is None:
            retries = int(os.environ.get("STOCK_FETCH_RETRIES") or "3")
        retries = max(1, retries)

        for attempt in range(retries):
            try:
                data = _fetch_quote_batch([code])
            except Exception:
                data = {}
            if data.get(code) is not None:
                pair = data[code]
                self._cache[code] = (now, pair)
                return pair
            if attempt < retries - 1:
                time.sleep(min(4.0, 0.5 * (2 ** attempt)))  # 0.5s, 1s, 2s ...
        return None

    def fetch(self, stock: Any) -> StockInputs:
        code = str(getattr(stock, "symbol", "")).strip().zfill(6)
        pair = self.fetch_one(code)
        if pair is None:
            raise RuntimeError(f"fallback 未返回 {code} 行情数据")
        price, pct = pair
        if price <= 0:
            raise RuntimeError(f"fallback 返回现价无效: {price}")

        # 1. 尝试拉取真实 K 线，计算技术指标
        symbol = _code_to_sina_symbol(code)
        klines = []
        if _akshare_available():
            klines = _fetch_akshare_kline(code)
        if not klines:
            klines = _fetch_sina_kline(symbol)

        if klines and len(klines) >= 14:
            technical = compute_real_technicals(klines)
            provenance_suffix = "kline"
        else:
            # K 线不可用，降级到涨跌幅代理指标
            technical = technical_from_spot_change(pct)
            provenance_suffix = "spot_proxy"

        fundamental = _neutral_fundamental(stock)
        # 情绪面：由真实技术指标构造（放量/缩量 + 均线排列 + 中期动量）
        sentiment = sentiment_from_technical(technical)
        # 板块/相对强度：fetch 时仅用当日涨跌幅作占位，
        # 真正的相对强度在引擎层用指数动量计算（见 predict_then_summarize._formula_predict）
        sector = sector_from_price_action(pct)
        return StockInputs(
            current_price=round(price, 2),
            change_percent=round(pct, 2),
            technical=technical,
            fundamental=fundamental,
            sentiment=sentiment,
            sector=sector,
            provenance=f"{'akshare_direct' if _akshare_available() else 'sina_http'}_{provenance_suffix}",
        )

# 全局单例
_global_akshare: Optional[AkshareFallbackProvider] = None


def get_akshare_provider() -> AkshareFallbackProvider:
    global _global_akshare
    if _global_akshare is None:
        _global_akshare = AkshareFallbackProvider()
    return _global_akshare
