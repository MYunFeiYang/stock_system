#!/usr/bin/env python3
"""
通过 OpenClaw Agent 使用其工具链中的「网页搜索 / 网页访问」拉取行情摘要。

依赖本机已配置好的 `openclaw` CLI 与模型（`openclaw agent --local` 或走 Gateway）。

环境变量：
  OPENCLAW_BIN                 openclaw 可执行文件（默认 PATH 中 openclaw）
  OPENCLAW_STOCK_AGENT_ID      默认 main
  OPENCLAW_AGENT_LOCAL         1=--local 嵌入式代理（默认）；0=走已运行的 Gateway
  OPENCLAW_AGENT_TIMEOUT       秒，默认 600（网页搜索较慢，可按需再加大）
  STOCK_OPENCLAW_CACHE_SEC     单股缓存秒数，默认 90
  STOCK_OPENCLAW_MAX_ATTEMPTS  单股最多调用 Agent 次数，默认 2（失败自动再试）
"""
from __future__ import annotations

import json
import os
import re
import shutil
import random
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

from data_providers import (
    StockInputs,
    _neutral_fundamental,
    sector_from_price_action,
    sentiment_from_price_action,
    technical_from_spot_change,
)


def _strip_json_from_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _parse_inner_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = _strip_json_from_text(text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(
        r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
        text,
        re.DOTALL,
    )
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _parse_agent_stdout(stdout: str) -> Tuple[Optional[str], str]:
    """返回 (payload_text, raw)。"""
    raw = stdout.strip()
    if not raw:
        return None, raw
    
    # Look for the JSON object containing payloads
    # Find all potential JSON objects and look for one with "payloads" key
    
    i = 0
    while i < len(raw):
        if raw[i] == '{':
            # Found opening brace, find matching closing brace
            brace_count = 1
            j = i + 1
            while j < len(raw) and brace_count > 0:
                if raw[j] == '{':
                    brace_count += 1
                elif raw[j] == '}':
                    brace_count -= 1
                j += 1
            
            if brace_count == 0:
                # Found a complete JSON object
                json_str = raw[i:j]
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict) and "payloads" in parsed:
                        payloads = parsed.get("payloads")
                        if isinstance(payloads, list) and payloads:
                            t = payloads[0].get("text")
                            if isinstance(t, str):
                                return t, raw
                except json.JSONDecodeError:
                    pass
                i = j
            else:
                i += 1
        else:
            i += 1
    
    # Fall back to original parsing
    try:
        outer = json.loads(raw)
    except json.JSONDecodeError:
        return raw, raw
    payloads = outer.get("payloads")
    if isinstance(payloads, list) and payloads:
        t = payloads[0].get("text")
        if isinstance(t, str):
            return t, raw
    return None, raw


def _find_openclaw_bin() -> str:
    """查找 openclaw 二进制路径，覆盖 nvm 等常见安装位置。"""
    # 1) 环境变量显式指定
    env_bin = os.environ.get("OPENCLAW_BIN")
    if env_bin and os.path.isfile(env_bin):
        return env_bin
    # 2) PATH 内查找
    found = shutil.which("openclaw")
    if found:
        return found
    # 3) 搜索 nvm 安装目录
    nvm_dir = os.environ.get(
        "NVM_DIR",
        os.path.join(os.path.expanduser("~"), ".nvm"),
    )
    for versions_dir in [os.path.join(nvm_dir, "versions", "node")]:
        if os.path.isdir(versions_dir):
            for ver in sorted(os.listdir(versions_dir), reverse=True):
                candidate = os.path.join(versions_dir, ver, "bin", "openclaw")
                if os.path.isfile(candidate):
                    # 同时把同级 node 加入 PATH 避免 shebang 解析失败
                    node_dir = os.path.join(versions_dir, ver, "bin")
                    existing = os.environ.get("PATH", "")
                    if node_dir not in existing:
                        os.environ["PATH"] = node_dir + os.pathsep + existing
                    return candidate
    # 4) 兜底
    return "openclaw"


class OpenclawAgentWebProvider:
    """
    调用 `openclaw agent`，由 Agent 使用 OpenClaw 配置的浏览/搜索类工具查询公开行情。
    技术面以涨跌幅粗估为主；基本面用板块中性基准（避免二次搜索爆炸）。

    当 openclaw agent 不可用时（gateway 挂了、超时等），自动降级到 akshare 直接拉取
    东方财富实时行情（环境变量 STOCK_USE_AKSHARE_FALLBACK 控制，默认开启）。
    """

    def __init__(self) -> None:
        self._bin = _find_openclaw_bin()
        self._agent_id = (os.environ.get("OPENCLAW_STOCK_AGENT_ID") or "main").strip()
        self._timeout = int(os.environ.get("OPENCLAW_AGENT_TIMEOUT") or "600")
        self._use_local = os.environ.get("OPENCLAW_AGENT_LOCAL", "1").lower() not in (
            "0",
            "false",
            "no",
        )
        self._cache: Dict[str, Tuple[float, Tuple[float, float]]] = {}
        self._ttl = float(os.environ.get("STOCK_OPENCLAW_CACHE_SEC") or "90")
        self._max_attempts = max(
            1, min(5, int(os.environ.get("STOCK_OPENCLAW_MAX_ATTEMPTS") or "2"))
        )
        self._use_fallback = os.environ.get("STOCK_USE_AKSHARE_FALLBACK", "1").lower() not in (
            "0", "false", "no",
        )
        self._fallback_available: Optional[bool] = None  # None=未检测, True/False

    def _check_gateway_health(self) -> bool:
        """快速检查 OpenClaw Gateway 是否可达（超时 3 秒）。"""
        if os.environ.get("STOCK_SKIP_AGENT", "").lower() in ("1", "true", "yes"):
            return False
        import urllib.request
        try:
            req = urllib.request.Request("http://127.0.0.1:18789/health")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _ensure_fallback(self) -> Optional[Any]:
        """延迟加载 akshare fallback provider（成功返回实例，失败返回 None）。"""
        if self._fallback_available is False:
            return None
        try:
            from akshare_fallback import get_akshare_provider
            p = get_akshare_provider()
            self._fallback_available = True
            return p
        except Exception:
            self._fallback_available = False
            return None

    def _invoke_agent(self, stock: Any, *, is_retry: bool = False) -> Dict[str, Any]:
        code = str(getattr(stock, "symbol", "")).strip().zfill(6)
        name = str(getattr(stock, "name", "")).strip()
        base = (
            f"任务：查询 A 股 {name}（股票代码 {code}）在公开网页上展示的「最新价/现价」"
            f"与「涨跌幅%」（相对前收的当日涨跌幅，例如 -1.25 表示跌 1.25%）。\n"
            f"请优先使用网页搜索找到 **东方财富、新浪财经、雪球、同花顺** 等主流行情页，"
            f"必要时用浏览器工具打开个股行情页，从行情 **表格或报价条** 读取数字，不要心算改写。\n"
            f"交易时段可参考「最新」；若已收盘可参考「收盘」价，仍以页面上展示为准。\n"
            f"务必基于实际打开的页面内容作答，查不到则填 null，不要编造。\n"
            f"最后只输出一行合法 JSON，不要 Markdown 代码块，格式严格为：\n"
            f'{{"current_price": <number|null>, "change_percent": <number|null>}}\n'
        )
        retry = ""
        if is_retry:
            retry = (
                "\n【重试】上次未返回有效 JSON 或 current_price 为 null。"
                "请直接打开 eastmoney 或 sina 财经上该股的行情页，从页面表格读取现价与涨跌幅后再输出上述一行 JSON。"
            )
        prompt = base + retry
        cmd: List[str] = [self._bin, "agent"]
        if self._use_local:
            cmd.append("--local")
        cmd.extend(
            [
                "--agent",
                self._agent_id,
                "--json",
                "-m",
                prompt,
                "--timeout",
                str(self._timeout),
            ]
        )
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout + 180,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError("openclaw agent 调用超时") from e
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()[:2000]
            raise RuntimeError(f"openclaw agent 失败 (code={proc.returncode}): {err}")
        
        # Try stdout first, then stderr (some OpenClaw versions output to stderr)
        payload_text, _ = _parse_agent_stdout(proc.stdout or "")
        if not payload_text:
            payload_text, _ = _parse_agent_stdout(proc.stderr or "")
        if not payload_text:
            raise RuntimeError("openclaw agent 返回无法解析（无 payloads.text）")
        low = payload_text.lower()
        if "timed out" in low or "timeoutseconds" in low.replace(" ", ""):
            raise RuntimeError(
                "OpenClaw Agent 在超时前未完成回复（未产出 JSON）。"
                "请增大环境变量 OPENCLAW_AGENT_TIMEOUT、在 openclaw.json 中提高 "
                "agents.defaults.timeoutSeconds，或换更快模型；"
                "带网页搜索的单股查询常需较长时间。"
            )
        
        # The payload_text should be a JSON string containing the stock data
        # First try to parse it as the outer JSON (from OpenClaw agent format)
        try:
            outer_data = json.loads(payload_text)
            if isinstance(outer_data, dict) and "text" in outer_data:
                # This is the OpenClaw agent format, extract the text field
                inner_text = outer_data["text"]
                if inner_text:
                    inner_data = json.loads(inner_text)
                    if isinstance(inner_data, dict) and "current_price" in inner_data:
                        return inner_data
        except json.JSONDecodeError:
            pass
        
        # If that fails, try direct parsing
        try:
            data = json.loads(payload_text)
            if isinstance(data, dict) and "current_price" in data:
                return data
        except json.JSONDecodeError:
            pass
        
        # If direct parsing fails, try the inner JSON extraction
        data = _parse_inner_json_object(payload_text)
        if not data:
            raise RuntimeError(f"无法从 Agent 回复中解析 JSON: {payload_text[:500]}")
        return data

    def fetch(self, stock: Any) -> StockInputs:
        code = str(getattr(stock, "symbol", "")).strip().zfill(6)
        now = time.time()
        if code in self._cache:
            ts, pair = self._cache[code]
            if now - ts < self._ttl:
                price, pct = pair
                return self._build_inputs(stock, price, pct)

        # ── 主路径：OpenClaw Agent ──
        gateway_ok = self._check_gateway_health()
        if gateway_ok:
            last_err: Optional[Exception] = None
            data: Optional[Dict[str, Any]] = None
            for att in range(self._max_attempts):
                try:
                    data = self._invoke_agent(stock, is_retry=(att > 0))
                    price_v = data.get("current_price")
                    if price_v is None:
                        raise RuntimeError(f"OpenClaw 未返回有效现价: {data!r}")
                    price = float(price_v)
                    if price <= 0:
                        raise RuntimeError(f"现价无效: {price}")
                    break
                except (RuntimeError, TypeError, ValueError) as e:
                    last_err = e
                    if att >= self._max_attempts - 1:
                        break  # 跌出循环走 fallback
                    time.sleep(1.5 + att * 1.5 + random.uniform(0, 0.6))

            if data is not None:
                price_v = data.get("current_price")
                chg_v = data.get("change_percent")
                try:
                    price = float(price_v)
                except (TypeError, ValueError):
                    price = 0.0
                try:
                    pct = float(chg_v) if chg_v is not None else 0.0
                except (TypeError, ValueError):
                    pct = 0.0
                if price > 0:
                    self._cache[code] = (now, (price, pct))
                    return self._build_inputs(stock, price, pct)

        # ── Fallback：akshare 直取 ──
        if self._use_fallback:
            fb = self._ensure_fallback()
            if fb is not None:
                print(f"  ⚠️ OpenClaw Agent 不可用，降级到 akshare 直取 {stock.name}({code})")
                return fb.fetch(stock)

        # ── 彻底失败 ──
        if gateway_ok and last_err:
            raise RuntimeError(
                f"单股拉价失败（OpenClaw 已试 {self._max_attempts} 次，"
                f"akshare fallback 不可用）: {last_err}"
            ) from last_err
        raise RuntimeError(
            "单股拉价失败：OpenClaw Gateway 不可达且 akshare fallback 不可用。"
            "请确认 gateway 已启动或安装 akshare。"
        )

    def _build_inputs(self, stock: Any, price: float, pct: float) -> StockInputs:
        technical = technical_from_spot_change(pct)
        fundamental = _neutral_fundamental(stock)
        sentiment = sentiment_from_price_action(pct, 1.0)
        sector = sector_from_price_action(pct)
        return StockInputs(
            current_price=round(price, 2),
            change_percent=round(pct, 2),
            technical=technical,
            fundamental=fundamental,
            sentiment=sentiment,
            sector=sector,
            provenance="openclaw_agent_web",
        )
