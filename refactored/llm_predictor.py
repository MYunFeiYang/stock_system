#!/usr/bin/env python3
"""
LLM 预测模块 — 通过 OpenClaw 的 openrouter 插件调用大模型进行股票预测。
（行情抓取由 akshare/新浪直连，无需 LLM；仅预测环节的 LLM 走 OpenClaw。）

机制：调用 `openclaw infer model run --model openrouter/<model> --prompt ... --json`，
走 OpenClaw gateway 路由（gateway 进程已通过 `openclaw models auth paste-api-key` 注入 openrouter 凭据，
模型路由由 OpenClaw 负责），脚本本身不持有 API Key。
注意：不用 --local（内嵌进程读不到 gateway 凭据）；调用时强制把 openclaw 同目录的 Node 置顶 PATH，
避免系统 Node 版本过低导致 openclaw CLI 拒绝启动。

环境变量：
  OPENROUTER_MODEL      模型名称（缺省跟随 OpenClaw 默认模型 openclaw.json→agents.defaults.model.primary；可裸 slug 或 openrouter/ 前缀覆盖）
  OPENROUTER_TIMEOUT    调用超时秒数（默认 30，gateway 路由调用已自动放宽）
  OPENCLAW_BIN          可选：openclaw 可执行文件路径（缺省自动探测 PATH / nvm）

依赖：Python 标准库 + 本地 openclaw CLI。LLM 调用失败时自动降级为纯公式打分。
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional


# ── 默认配置 ──────────────────────────────────────────
_DEFAULT_MODEL = "deepseek/deepseek-chat"
_DEFAULT_TIMEOUT = 30

# openclaw CLI 候选路径（按优先级：PATH > 已知 nvm 安装）
_OPENCLAW_CANDIDATES = [
    "~/.nvm/versions/node/v24.19.0/bin/openclaw",
    "~/.nvm/versions/node/*/bin/openclaw",
]


def _find_openclaw_bin() -> Optional[str]:
    """定位 openclaw CLI。

    优先使用已知 nvm 安装路径中的 Node>=24 版本（openclaw CLI 要求
    Node>=22.22.3/<23 或 >=24.15，系统自带的低版本会直接拒绝启动），
    其次回退 PATH。"""
    for c in _OPENCLAW_CANDIDATES:
        c = os.path.expanduser(c)
        if "*" in c:
            for g in sorted(glob.glob(c), reverse=True):
                if os.path.exists(g):
                    return g
        elif os.path.exists(c):
            return c
    return shutil.which("openclaw") or None


def _get_openclaw_default_model() -> str:
    """读取 OpenClaw 全局默认模型（openclaw.json → agents.defaults.model.primary）。
    这样股票系统自动跟随你在 OpenClaw 里切换的模型；找不到时回退内置默认。"""
    cfg_path = os.environ.get("OPENCLAW_CONFIG") or os.path.expanduser("~/.openclaw/openclaw.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        primary = (
            cfg.get("agents", {})
            .get("defaults", {})
            .get("model", {})
            .get("primary")
        )
        if isinstance(primary, str) and primary.strip():
            return primary.strip()
    except Exception:
        pass
    return _DEFAULT_MODEL


def _get_model() -> str:
    """模型名解析优先级：OPENROUTER_MODEL 环境变量 > OpenClaw 默认模型 > 内置默认。
    自动补 openrouter/ 前缀（兼容裸 slug 与 openrouter/ 前缀两种写法）。"""
    m = os.environ.get("OPENROUTER_MODEL") or _get_openclaw_default_model()
    return m if m.startswith("openrouter/") else f"openrouter/{m}"


def _get_timeout() -> int:
    try:
        return int(os.environ.get("OPENROUTER_TIMEOUT", str(_DEFAULT_TIMEOUT)))
    except ValueError:
        return _DEFAULT_TIMEOUT


# ── Prompt 模板 ────────────────────────────────────────

_SYSTEM_PROMPT = """你是一位资深A股量化分析师。你需要根据提供的股票多维度数据，给出客观的短期（1-5个交易日）走势预测。

要求：
1. 重点分析技术面趋势结构：均线排列（多头/空头/缠绕）、MACD状态、RSI超买超卖、布林带位置、量能变化
2. 结合基本面估值与行业轮动状态
3. 评分 1-10 分（10=强烈看涨，1=强烈看跌，5=中性），必须基于多周期趋势一致性给出，不要仅看单日涨跌
4. 信号等级：强烈买入(>=8.5)、买入(>=7.0)、持有(5.0-7.0)、卖出(3.5-5.0)、强烈卖出(<3.5)
5. 信心度 40-95，反映你对判断的确定性（趋势越清晰信心越高）
6. 给出 2-4 条简短的推荐理由（每条不超过20字），须具体（如"量价齐升"而非"技术面好"）

只输出 JSON，不要其他内容。格式：
{"final_score": 7.5, "signal": "买入", "confidence": 72, "reasons": ["均线多头排列", "MACD金叉放量"]}"""


def _build_single_stock_prompt(stock: Dict[str, Any]) -> str:
    """构建单只股票的预测 prompt"""
    name = stock.get("name", "未知")
    symbol = stock.get("symbol", "")
    sector = stock.get("sector_name", stock.get("sector", ""))
    if isinstance(sector, dict):
        sector = sector.get("name", "")
    price = stock.get("current_price", 0)
    change = stock.get("change_percent", 0)
    tech = stock.get("technical", {})
    fund = stock.get("fundamental", {})
    sent = stock.get("sentiment", {})
    sect = stock.get("sector", {}) if isinstance(stock.get("sector"), dict) else {}

    lines = [
        f"请预测以下A股未来1-5个交易日的走势：",
        f"",
        f"股票：{name}（{symbol}）",
        f"行业：{sector}",
        f"现价：{price} 元 | 今日涨跌：{change:+.2f}%",
        f"",
        f"【技术面 - 多周期趋势】",
        f"  RSI(14): {tech.get('rsi', 50)}",
        f"  MACD: {tech.get('macd_signal', '中性')} (DIF={tech.get('macd_dif', 0)}, DEA={tech.get('macd_dea', 0)}, BAR={tech.get('macd_bar', 0)})",
        f"  布林带: 位置={tech.get('bollinger_position', 0)} (上={tech.get('bollinger_upper')}, 中={tech.get('bollinger_mid')}, 下={tech.get('bollinger_lower')})",
        f"  量比: {tech.get('volume_ratio', 1.0)}",
        f"  均线排列: {tech.get('ma_alignment', '中性')}",
        f"  均线值: MA5={tech.get('ma5')}, MA10={tech.get('ma10')}, MA20={tech.get('ma20')}, MA60={tech.get('ma60')}",
        f"  动量: 5日={tech.get('momentum_5d', 0):+.2%}, 10日={tech.get('momentum_10d', 0):+.2%}, 20日={tech.get('momentum_20d', 0):+.2%}",
        f"",
        f"【基本面】",
        f"  PE: {fund.get('pe_ratio', 'N/A')} | PB: {fund.get('pb_ratio', 'N/A')} | ROE: {fund.get('roe', 'N/A')}%",
        f"  营收增速: {fund.get('growth_rate', 'N/A')}%",
        f"  资产负债率: {fund.get('debt_ratio', 'N/A')}",
        f"  股息率: {fund.get('dividend_yield', 'N/A')}%",
        f"",
        f"【情绪面】",
        f"  市场热度: {sent.get('market_heat', 5)}/10",
        f"  机构关注: {sent.get('institution_attention', 5)}/10",
        f"  散户情绪: {sent.get('retail_sentiment', '中性')}",
        f"  新闻面: {sent.get('news_sentiment', '中性')}",
        f"",
        f"【行业面】",
        f"  景气度: {sect.get('prosperity', 5)}/10",
        f"  政策支持: {sect.get('policy_support', 5)}/10",
        f"  资金流向: {sect.get('capital_flow', 5)}/10",
        f"  轮动位置: {sect.get('rotation_position', 5)}/10",
    ]
    return "\n".join(lines)


def _parse_response(text: str) -> Optional[Dict[str, Any]]:
    """从 LLM 响应中解析 JSON"""
    text = text.strip()
    # 免费模型(openrouter/free)偶发返回安全包装前缀（如 "User Safety: safe"、
    # "Assistant:"）而非纯 JSON。剥离这些前缀后再解析；若剥离后仍无 JSON，
    # 则视为拒绝/无法解析，上层会降级为公式打分。
    _safety_prefix = re.compile(
        r"^\s*(user\s*safety\s*:\s*safe|safety\s*:\s*safe|assistant\s*:)\s*",
        re.IGNORECASE,
    )
    m = _safety_prefix.match(text)
    if m:
        text = text[m.end():].strip()
    # 尝试提取 JSON 块
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # 尝试找第一个 { }
        try:
            s = text.index("{")
            e = text.rindex("}") + 1
            result = json.loads(text[s:e])
        except (ValueError, json.JSONDecodeError):
            return None

    # 校验必要字段
    required = {"final_score", "signal", "confidence", "reasons"}
    if not required.issubset(result.keys()):
        return None

    # 类型校验
    try:
        result["final_score"] = float(result["final_score"])
        result["confidence"] = int(result["confidence"])
        result["signal"] = str(result["signal"])
        result["reasons"] = list(result["reasons"])
    except (ValueError, TypeError):
        return None

    # 范围校验
    result["final_score"] = max(1.0, min(10.0, result["final_score"]))
    result["confidence"] = max(30, min(95, result["confidence"]))
    valid_signals = {"强烈买入", "买入", "持有", "卖出", "强烈卖出"}
    if result["signal"] not in valid_signals:
        # 尝试映射英文
        signal_map = {
            "strong buy": "强烈买入", "buy": "买入",
            "hold": "持有", "sell": "卖出", "strong sell": "强烈卖出",
        }
        result["signal"] = signal_map.get(result["signal"].lower(), "持有")

    return result


def _extract_content_from_openclaw_output(text: str) -> Optional[str]:
    """从 `openclaw infer model run --json` 的输出中抽出模型回复文本。

    兼容两种 --json 结构：
      - 官方封装：`{"ok": true, "outputs": [{"text": "..."}], ...}`
      - 原生 OpenAI：`{"choices": [{"message": {"content": "..."}}]}`
    非 JSON 时把整段 stdout 当作回复。
    返回 None 表示调用失败（ok=false / 含 error / 无内容）。"""
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return text  # CLI 直接输出纯文本（如我们的预测 JSON）
    if not isinstance(obj, dict):
        return text

    # 失败信号：ok=false 或含 error 字段
    if obj.get("ok") is False or obj.get("error") or obj.get("status") == "error":
        return None

    # 官方封装结构：outputs[].text
    outputs = obj.get("outputs")
    if isinstance(outputs, list) and outputs:
        first = outputs[0]
        if isinstance(first, dict):
            t = first.get("text")
            if isinstance(t, str) and t.strip():
                return t
        elif isinstance(first, str) and first.strip():
            return first

    # 通用顶层字段
    for key in ("content", "text", "output", "response", "result"):
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            return v

    # 原生 OpenAI 结构：choices[0].message.content
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        c = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(c, str) and c.strip():
            return c
    return None


def call_openrouter(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """通过 OpenClaw 的 openrouter 插件调用 LLM，返回与 OpenRouter API 同构的
    `{"choices":[{"message":{"content":...}}], "model":...}` 以便上层逻辑不变。
    失败（CLI 缺失/超时/非零退出/输出无法解析）返回 None，触发公式降级。"""
    bin_path = os.environ.get("OPENCLAW_BIN") or _find_openclaw_bin()
    if not bin_path:
        print("[LLM] 未找到 openclaw CLI，跳过 LLM 预测（仅用公式打分）")
        return None

    # CLI 以 --prompt 传单段文本：合并 system/user 角色
    prompt_text = "\n\n".join(
        m.get("content", "") for m in messages if m.get("content")
    )
    model_ref = model or _get_model()  # 已是 openrouter/xxx 格式
    if not model_ref.startswith("openrouter/"):
        model_ref = f"openrouter/{model_ref}"

    cmd = [
        bin_path, "infer", "model", "run",
        "--model", model_ref,
        "--prompt", prompt_text,
        "--json",
    ]
    # 走 gateway 路由（gateway 进程已注入 openrouter 凭据）；
    # 内嵌 --local 模式读不到 gateway 的凭据，故不用 --local。
    # 强制把 openclaw 同目录的 Node 置顶 PATH，避免 env node 解析到过低版本。
    env = dict(os.environ)
    node_bin_dir = os.path.dirname(bin_path)
    env["PATH"] = node_bin_dir + os.pathsep + env.get("PATH", "")
    timeout = _get_timeout() + 30
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        print(f"[LLM] openclaw infer 超时（{timeout}s）")
        return None
    except Exception as e:
        print(f"[LLM] openclaw infer 调用失败: {e}")
        return None

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip()
        print(f"[LLM] openclaw infer 返回非零({proc.returncode}): {err[:300]}")
        return None

    content = _extract_content_from_openclaw_output(proc.stdout)
    if not content:
        print(f"[LLM] openclaw infer 输出为空: {proc.stdout[:300]}")
        return None

    return {"choices": [{"message": {"content": content}}], "model": model_ref}


def predict_single(stock: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """对单只股票调用 LLM 预测，失败返回 None"""
    user_prompt = _build_single_stock_prompt(stock)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    resp = call_openrouter(messages)
    if not resp:
        return None

    try:
        content = resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        print("[LLM] 响应格式异常")
        return None

    result = _parse_response(content)
    if result:
        result["_model"] = resp.get("model", "unknown")
        print(f"[LLM] {stock.get('name', '?')} → {result['signal']}({result['final_score']}) "
              f"信心{result['confidence']} [{result['_model']}]")
    else:
        # 常见原因：免费模型返回 "User Safety: safe" 安全包装且无 JSON 内容
        hint = "（安全包装/无JSON，降级公式）" if re.search(r"user\s*safety", content, re.I) else ""
        print(f"[LLM] {stock.get('name', '?')} 响应解析失败{hint}: {content[:200]}")
    return result


# ── 便捷入口：与现有 PredictionEngine 对接 ──

def stock_to_llm_input(
    stock_name: str,
    stock_symbol: str,
    stock_sector: str,
    current_price: float,
    change_percent: float,
    technical: Dict,
    fundamental: Dict,
    sentiment: Dict,
    sector: Dict,
) -> Dict[str, Any]:
    """将 StockConfig + StockInputs 组装成 llm_predictor 需要的字典"""
    return {
        "name": stock_name,
        "symbol": stock_symbol,
        "sector_name": stock_sector,
        "current_price": current_price,
        "change_percent": change_percent,
        "technical": technical,
        "fundamental": fundamental,
        "sentiment": sentiment,
        "sector": sector,
    }


def is_llm_enabled() -> bool:
    """检查 LLM 预测是否可用：依赖 OpenClaw CLI（其 openrouter 插件提供 key 与路由）"""
    return _find_openclaw_bin() is not None
