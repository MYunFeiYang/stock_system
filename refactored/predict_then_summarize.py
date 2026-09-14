#!/usr/bin/env python3
"""
A股分析系统 - 符合"先预测再总结"理念的重构版本
模块化设计，清晰的预测->总结流程
"""

import json
import os
import sys
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from pathlib import Path

from data_providers import StockDataProvider, get_analysis_type_name, get_default_provider, SECTOR_BENCHMARKS
from evening_optimizer import EveningPredictionOptimizer
from reconcile_accuracy import default_accuracy_tuning, merge_accuracy_tuning
from llm_predictor import (
    predict_single, stock_to_llm_input, is_llm_enabled,
)


def _default_stock_system_root() -> str:
    return os.environ.get(
        "STOCK_SYSTEM_ROOT",
        str(Path(__file__).resolve().parent.parent),
    )


def _alpha_evidence_disclaimer() -> str:
    """早盘风险提示：动态引用 walk-forward 超额收益(alpha)诊断结论。

    为什么改成动态：原实现把样本量与 t 值**硬编码**在文案里（"仅3个交易日/22笔
    买入""隔日 t=-1.76"），随样本积累这些数字会腐烂，导致报告长期播报过期数据，
    且措辞停留在"既不能证明有效也不能证明无效"——而大样本基准对齐检验已能给出
    **证伪**结论。硬编码的免责声明本身就是一种误导。

    数据源: data/alpha_check_report.json（由 refactored/walkforward_alpha_check.py 生成）
    失败回退: 不含任何具体数字的保守措辞（宁可少说，不可说错）。
    """
    try:
        p = Path(_default_stock_system_root()) / "data" / "alpha_check_report.json"
        if not p.exists():
            raise FileNotFoundError(str(p))
        d = json.loads(p.read_text(encoding="utf-8"))
        g = {x["label"]: x for x in (d.get("groups") or [])}
        buy = g.get("全部买入(原始公式)")
        if not buy:
            raise KeyError("全部买入(原始公式)")
        days = (d.get("range") or {}).get("trading_days") or "历史"
        head = (
            "⚠️ 风险提示：以下为公式信号的日常观察，非买卖建议。"
            f"个股选择在 {days} 交易日 walk-forward 回放 + 基准对齐检验中"
            f"【未检出可用 alpha】：买入组绝对收益 {buy['abs_mean_pct']:+.3f}%/日，"
            f"同期基准 {buy['bench_mean_pct']:+.3f}%/日，"
            f"超额 alpha 仅 {buy['alpha_mean_pct']:+.3f}%/日，"
            f"t={buy['alpha_t']:+.2f} 不显著。"
        )
        up = g.get("买入·大盘上涨日")
        if up:
            head += (
                f"此前『大盘上涨日显著盈利』经基准对齐后实为跑输：组合 "
                f"{up['abs_mean_pct']:+.3f}%/日 vs 指数 {up['bench_mean_pct']:+.3f}%/日"
                f"（alpha {up['alpha_mean_pct']:+.3f}%/日）——那是市场 beta，不是选股能力。"
            )
        head += _power_caveat()
        head += _timing_and_gate_evidence()
        head += "系统定位为核心池观察 feed，请勿据此实盘操作，仅作研究参考。"
        return head
    except Exception:
        # 宁可少说，不可说错：拿不到证据时不写任何具体数字。
        return (
            "⚠️ 风险提示：以下为公式信号的日常观察，非买卖建议。"
            "个股选择尚未取得经基准对齐验证的超额收益，"
            "系统定位为核心池观察 feed，请勿据此实盘操作，仅作研究参考。"
        )


def _alpha_evidence_disclaimer_short() -> str:
    """企微推送用的精简免责声明（受 2048 字节硬限约束，必须短）。

    只给结论，不给数字；详情在完整报告文件里。
    """
    try:
        d = _read_json("alpha_check_report.json")
        if not d:
            raise FileNotFoundError()
        return ("⚠️ 观察feed(非建议)：个股选股alpha经基准对齐检验【未检出】；"
                "大盘择时/路线A闸门均无证据支持。请勿据此操作，完整证据见报告文件。")
    except Exception:
        return ("⚠️ 观察feed(非建议)：系统无可用选股edge，仅作行情与相对强弱观察，"
                "请勿据此实盘操作。")


def _read_json(path: str):
    """读项目 data/ 下的 JSON 报告；缺失或损坏一律返回 None（调用方降级）。"""
    try:
        p = Path(_default_stock_system_root()) / "data" / path
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _power_caveat() -> str:
    """统计功效声明：说清"检不出"与"不存在"的区别。

    为什么必须加: 拍板 #002 曾断言"不是样本不足"。功效分析推翻了这一说法——
    本样本(20 只 × 239 天)的最小可检出效应高达年化 38%，现实规模的 alpha
    (年化 5~15%)在统计上根本检不出。把"未检出"说成"已证伪"是过度声明，
    同样是一种误导。诚实表述 = 未检出 + 灵敏度上限。
    """
    d = _read_json("power_check_report.json")
    if not d:
        return ""
    try:
        mde_long = d["mde_long_pct_day"] * 240 / 2      # 折算年化
        mde_ls = d["mde_ls_pct_day"] * 240 / 2
        return (f"注意灵敏度上限：本样本仅能检出年化 ≥{mde_long:.0f}% 的 alpha"
                f"（多空口径 ≥{mde_ls:.0f}%），"
                f"更小的 alpha 无法检出——未检出不等于不存在。")
    except Exception:
        return ""


def _timing_and_gate_evidence() -> str:
    """大盘择时 + 路线 A 闸门的证据陈述（拍板 #003 补入）。

    为什么改：拍板 #002 曾写「路线 A 仅作回撤风控保留」，该说法已被直接验证
    推翻——被闸门压掉的买入 alpha 与 0 无差异，压掉它没有依据。此处改为动态
    引用 gate_check / index_timing 报告，禁止报告继续播报已被证伪的结论。
    """
    parts = []

    d = _read_json("index_timing_report.json")
    if d and d.get("verdict"):
        try:
            win = sum(int(x["win_rate"].split("/")[0]) for x in d["verdict"])
            tot = sum(int(x["win_rate"].split("/")[1]) for x in d["verdict"])
            med = min(x["median_excess"] for x in d["verdict"])
            parts.append(
                f"大盘择时同样无 edge：12 年、3 个宽基指数 × {tot} 组参数，"
                f"仅 {win} 组跑赢买入持有（中位超额最差 {med:+.2f}%/年），"
                f"零成本下结论不变——不是成本问题，是策略本身无效"
            )
        except Exception:
            pass

    g = _read_json("gate_check_report.json")
    if g and g.get("groups"):
        blk = next((x for x in g["groups"]
                    if x["label"] == "闸门压掉(实际不买)"), None)
        if blk:
            parts.append(
                "路线 A 大盘闸门无证据支持：被它压掉的买入超额 "
                f"{blk['alpha_pct']:+.3f}%/日 t={blk['t']:+.2f}"
                f"{'（不显著，与 0 无差异）' if not blk['significant'] else ''}"
                "，压掉它没有依据"
            )

    if not parts:
        return ""
    return "；".join(parts) + "。"


# ==================== 核心模型 ====================

class SignalType(Enum):
    STRONG_BUY = "强烈买入"
    BUY = "买入" 
    HOLD = "持有"
    SELL = "卖出"
    STRONG_SELL = "强烈卖出"


def apply_signal_margin(signal: str, final_score: float, th: Dict[str, float], margin: float) -> str:
    """
    阈值边界保守化：贴边的买入/卖出/强档略向持有或弱一档收拢，减少勉强信号。
    margin<=0 时不调整。
    """
    if margin <= 0:
        return signal
    sb = float(th["strong_buy"])
    b = float(th["buy"])
    h = float(th["hold"])
    s = float(th["sell"])
    if signal == SignalType.BUY.value and b <= final_score < b + margin:
        return SignalType.HOLD.value
    if signal == SignalType.STRONG_BUY.value and sb <= final_score < sb + margin:
        return SignalType.BUY.value
    if signal == SignalType.SELL.value and h - margin < final_score < h:
        return SignalType.HOLD.value
    if signal == SignalType.STRONG_SELL.value and s <= final_score < s + margin:
        return SignalType.SELL.value
    return signal


@dataclass
class StockConfig:
    """股票基础配置"""
    name: str
    symbol: str
    sector: str
    weight: float


@dataclass
class PredictionResult:
    """预测结果"""
    stock: StockConfig
    current_price: float
    change_percent: float
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    sector_score: float
    final_score: float
    signal: str
    confidence: int
    reasons: List[str]
    prediction_time: datetime
    data_provenance: str = "openclaw_agent_web"
    formula_signal: str = None  # A/B 对照：纯公式信号（LLM 开启时与 signal 区分）
    
    def to_dict(self) -> Dict:
        return {
            'stock': asdict(self.stock),
            'current_price': self.current_price,
            'change_percent': self.change_percent,
            'technical_score': self.technical_score,
            'fundamental_score': self.fundamental_score,
            'sentiment_score': self.sentiment_score,
            'sector_score': self.sector_score,
            'final_score': self.final_score,
            'signal': self.signal,
            'formula_signal': self.formula_signal,
            'confidence': self.confidence,
            'reasons': self.reasons,
            'prediction_time': self.prediction_time.isoformat(),
            'data_provenance': self.data_provenance,
        }


@dataclass
class SummaryReport:
    """总结报告"""
    report_time: datetime
    analysis_type: str  # 'morning', 'afternoon', 'evening', 'weekly'
    buy_recommendations: List[PredictionResult]
    sell_recommendations: List[PredictionResult]
    hold_recommendations: List[PredictionResult]
    market_overview: Dict
    sector_analysis: Dict
    risk_alerts: List[str]
    next_actions: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'report_time': self.report_time.isoformat(),
            'analysis_type': self.analysis_type,
            'buy_recommendations': [pred.to_dict() for pred in self.buy_recommendations],
            'sell_recommendations': [pred.to_dict() for pred in self.sell_recommendations],
            'hold_recommendations': [pred.to_dict() for pred in self.hold_recommendations],
            'market_overview': self.market_overview,
            'sector_analysis': self.sector_analysis,
            'risk_alerts': self.risk_alerts,
            'next_actions': self.next_actions
        }


def _prev_close_anchor(symbol: str):
    """取早盘预测的锚定基准：(上一交易日收盘, 上一交易日相对前一交易日的真实涨跌幅%)。

    返回 None 表示取不到历史（非交易日/无数据），调用方应回退到实时价。
    """
    from akshare_fallback import fetch_recent_ohlc
    rec = fetch_recent_ohlc(symbol, n=2)
    if not rec or len(rec) < 1:
        return None
    last_open, last_close = rec[-1]
    prev_close = last_close
    if len(rec) >= 2:
        _po, prev_prev_close = rec[-2]
        chg = (last_close - prev_prev_close) / prev_prev_close * 100.0 if prev_prev_close > 0 else 0.0
    else:
        # 仅 1 天可用：用当日日内涨跌近似
        chg = (last_close - last_open) / last_open * 100.0 if last_open > 0 else 0.0
    return (prev_close, round(chg, 2))


def _use_llm() -> bool:
    """是否启用 LLM 预测（环境变量 STOCK_USE_LLM=1）"""
    return os.environ.get("STOCK_USE_LLM") == "1" and is_llm_enabled()


# ==================== 预测引擎 ====================

def _load_low_confidence_symbols() -> Dict[str, float]:
    """读 data/iteration_briefing.txt 的【需加严复核的标的】段，
    提取近期方向多次不一致的标的及其方向不一致率(miss_rate=M/N)。
    用于早盘低置信降权（股神风控：不跟已知会错的信号硬刚）。
    无文件/无段时返回空 dict（不影响正常预测）。"""
    import re
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "data" / "iteration_briefing.txt"
    if not p.exists():
        return {}
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        return {}
    lines = text.splitlines()
    in_section = False
    result: Dict[str, float] = {}
    # 仅匹配段内 "近N次中不一致M次" 格式，避免误抓第6行概览
    sym_re = re.compile(r'近(\d+)次中不一致\s*(\d+)\s*次')
    for line in lines:
        if '需加严复核的标的' in line:
            in_section = True
            continue
        if in_section:
            if not line.strip() or line.strip().startswith('说明'):
                break
            code_m = re.search(r'\((\d{6})\)', line)
            m = sym_re.search(line)
            if code_m and m:
                n = int(m.group(1)); miss = int(m.group(2))
                result[code_m.group(1)] = round(miss / n, 3) if n > 0 else 1.0
    return result


class PredictionEngine:
    """预测引擎 - 核心预测逻辑"""
    
    def __init__(self, data_provider: Optional[StockDataProvider] = None):
        self._provider = data_provider or get_default_provider()
        self.scoring_engine = ScoringEngine()
        self.signal_generator = SignalGenerator()
        self._use_llm = _use_llm()
        self._llm_consec_fail = 0  # LLM 连续失败计数：达阈值整批降级纯公式
        self._market_status: Optional[Dict] = None
        self._market_regime: str = "range"  # 默认震荡市（中性）
        self._index_closes: Optional[List[float]] = None  # 上证指数收盘序列（路线A闸门用）
        self._market_gate: str = "normal"  # 路线A: down_trend / normal
        self._skipped_stocks: List[str] = []  # 行情源瞬时失败未纳入的个股（报告正文标注用）

    def _detect_market_status(self) -> Dict:
        """检测当前市场状态（基于上证指数 K 线，缓存避免重复请求）。"""
        if self._market_status is not None:
            return self._market_status
        default_status = {
            "status": "unknown", "regime": "unknown", "volatility": None,
            "trend_strength": None, "bollinger_bandwidth": None,
            "description": "市场状态检测失败（行情源瞬时抖动），状态未知（非震荡市，勿据此判断）",
        }
        try:
            from akshare_fallback import fetch_index_klines
            from data_providers import detect_market_status
            # 多取 5 根(65)供 60DMA 平滑；内部 regime 仍用最后 60 根，行为不变
            klines = fetch_index_klines("sh000001", 65)
            if klines and len(klines) >= 60:
                status = detect_market_status(klines[-60:])
                self._index_closes = [float(k.get("close") or 0) for k in klines if k.get("close")]
            else:
                status = default_status
        except Exception:
            status = default_status
        self._market_status = status
        self._market_regime = status.get("regime", "range")
        return status

    def _detect_index_trend_gate(self) -> str:
        """路线 A 大盘趋势闸门（股神 8-24 批准）：干净的上证 20/60 DMA 多空信号。

        明确下跌趋势 = 收盘 < 60DMA 且 20DMA < 60DMA → 'down_trend'；否则 'normal'。
        绝不用系统内部 regime（其 trending_up 反预示亏损，walk-forward 已证滞后/追高）。
        取数失败/数据不足(<60 根) → 'normal'（fail-open，不阻断现有行为）。"""
        closes = self._index_closes
        if not closes or len(closes) < 60:
            return "normal"
        try:
            last = closes[-1]
            ma20 = sum(closes[-20:]) / 20.0
            ma60 = sum(closes[-60:]) / 60.0
            if last < ma60 and ma20 < ma60:
                return "down_trend"
        except Exception:
            return "normal"
        return "normal"

    def _apply_market_trend_gate(self, results: List[PredictionResult]) -> None:
        """大盘/regime 双闸门：
          - 路线 A（股神 8-24）：上证 20/60DMA 明确下跌趋势 → 压买入为持有（仅停多，不开空）
          - 追高闸门（股神 9-03，OOS 验证 walkforward_oos_report.json）：指数
            regime=trend/volatility 时公式买入显著追高亏损（训练期 trend 买入
            -0.85%/日 t=-2.07），压为持有。route-A 的 20/60DMA 只覆盖下跌趋势，
            漏掉 trending_up 上涨追高，此闸门补该缺口。
        降级同时压 final_score 到买入线以下，使报告统计与信号一致。
        fail-open：取数失败/regime=range（默认）不改动，向后兼容。"""
        gate = self._detect_index_trend_gate()      # "down_trend" | "normal"
        regime = self._market_regime                 # "range" | "trend" | "volatility"
        self._market_gate = gate

        suppress = False
        reason = ""
        tag = ""
        if gate == "down_trend":
            suppress = True
            reason = "大盘下跌趋势闸门:买入信号暂停(仅持有, 不跟跌)"
            tag = "大盘闸门"
        elif regime in ("trend", "volatility"):
            # 追高闸门：趋势/高波动市公式买入易追高，walk-forward 证伪其有负 alpha
            suppress = True
            label = "趋势" if regime == "trend" else "高波动"
            reason = f"追高闸门:指数处{label}市,公式买入易追高,暂停(仅持有)"
            tag = "追高闸门"

        if not suppress:
            return
        try:
            b_line = ConfigManager.get_signal_thresholds().get("buy", 5.8)
        except Exception:
            b_line = 5.8
        for r in results:
            if r.signal in (SignalType.BUY.value, SignalType.STRONG_BUY.value):
                old = r.signal
                r.final_score = max(1.0, round(b_line - 0.1, 2))
                r.signal = SignalType.HOLD.value
                r.reasons.append(reason)
                print(f"[{tag}] {r.stock.name}({r.stock.symbol}) {old}→持有 (regime={regime})")
    
    def _blend_with_llm(self, stock: StockConfig, inputs, formula_result: PredictionResult, llm_r: dict) -> PredictionResult:
        """
        LLM 叠加层：将 LLM 判断与公式打分融合，取加权平均。
        
        融合策略：
        - formula 提供一致性和纪律性
        - LLM 提供模式识别和语义理解
        - 两者差异过大时（>2.5分）降低信心度，提示不确定性
        """
        formula_score = formula_result.final_score
        llm_score = float(llm_r["final_score"])

        # 加权平均：公式 60% + LLM 40%
        blended = formula_score * 0.6 + llm_score * 0.4
        blended = max(1.0, min(10.0, blended))

        # 判定信号
        signal, conf, reasons = self.signal_generator.generate_signal(blended, stock, inputs.technical)

        # 信心度：取两者中较低的，且差异大时进一步打折
        diff = abs(formula_score - llm_score)
        base_conf = min(formula_result.confidence, int(llm_r["confidence"]))
        if diff > 2.5:
            base_conf = int(base_conf * 0.8)  # 分歧大 → 降低信心
        conf = base_conf

        # 理由：合并公式 + LLM（去重，最多 4 条）
        formula_reasons = list(formula_result.reasons or [])
        llm_reasons = list(llm_r.get("reasons", []))
        merged = []
        seen = set()
        for r in llm_reasons + formula_reasons:
            if r and r not in seen:
                seen.add(r)
                merged.append(r)
        merged = merged[:4]

        return PredictionResult(
            stock=stock,
            current_price=inputs.current_price,
            change_percent=inputs.change_percent,
            technical_score=formula_result.technical_score,
            fundamental_score=formula_result.fundamental_score,
            sentiment_score=formula_result.sentiment_score,
            sector_score=formula_result.sector_score,
            final_score=round(blended, 2),
            signal=signal,
            confidence=conf,
            reasons=merged,
            prediction_time=datetime.now(),
            data_provenance=f"formula+llm:{llm_r.get('_model', 'unknown')}",
        )

    def _try_llm_predict(self, stock: StockConfig, inputs) -> Optional[dict]:
        """尝试用 LLM 预测单只股票，返回原始 dict 或 None"""
        llm_input = stock_to_llm_input(
            stock_name=stock.name,
            stock_symbol=stock.symbol,
            stock_sector=stock.sector,
            current_price=inputs.current_price,
            change_percent=inputs.change_percent,
            technical=inputs.technical,
            fundamental=inputs.fundamental,
            sentiment=inputs.sentiment,
            sector=inputs.sector,
        )
        return predict_single(llm_input)

    def predict_stock(self, stock: StockConfig, analysis_type: str = "evening") -> PredictionResult:
        """对单只股票进行预测：先算公式，再叠加 LLM（若可用）"""
        
        self._detect_market_status()
        inputs = self._provider.fetch(stock)

        # ── 早盘锚定上一交易日收盘：让预测与 cron 实际运行时刻解耦 ──
        # 否则机器睡眠导致早盘 cron 延后到盘中跑时，会用盘中实时价做基线，
        # "早报"名不副实。改用已收盘的历史值后，08:00 或盘中补跑结果完全一致。
        if analysis_type == "morning":
            anchor = _prev_close_anchor(stock.symbol)
            if anchor:
                prev_close, prev_change = anchor
                inputs.current_price = round(prev_close, 2)
                inputs.change_percent = prev_change
                from data_providers import (
                    sector_from_price_action,
                    technical_from_spot_change,
                    sentiment_from_technical,
                )
                inputs.sector = sector_from_price_action(prev_change)
                if inputs.provenance.endswith("spot_proxy"):
                    # K 线不可用、technical 由实时涨跌幅派生时才需重算；
                    # 真实 K 线算出的 technical/sentiment 本身是历史值、已与时刻无关，保留。
                    inputs.technical = technical_from_spot_change(prev_change)
                    inputs.sentiment = sentiment_from_technical(inputs.technical)
                inputs.provenance = inputs.provenance + "_prevclose_anchor"

        # ── 公式打分（始终计算，A/B 对照的纯公式基线）──
        formula_result = self._formula_predict(stock, inputs, analysis_type)

        # ── LLM 叠加（若启用且可用）──
        if self._use_llm:
            llm_r = self._try_llm_predict(stock, inputs)
            if llm_r:
                self._llm_consec_fail = 0
                blended = self._blend_with_llm(stock, inputs, formula_result, llm_r)
                blended.formula_signal = formula_result.signal  # A/B：记录纯公式信号
                return blended
            # 单只失败：计数 + 连续失败达阈值则整批降级，避免 20 只全卡/主进程 OOM
            self._llm_consec_fail += 1
            if self._llm_consec_fail >= 3:
                self._use_llm = False
                print("[LLM] 连续失败 3 次，后续整批降级为纯公式打分（避免拖死/OOM）")
            else:
                print(f"[LLM] {stock.name} 预测失败，仅用公式打分")
        formula_result.formula_signal = formula_result.signal  # 公式路径同样记录纯公式信号
        return formula_result
    
    def _formula_predict(self, stock: StockConfig, inputs, analysis_type: str) -> PredictionResult:
        """纯公式打分（LLM 失败时的 fallback）"""
        technical_score = self.scoring_engine.calculate_technical_score(inputs.technical)
        fundamental_score = self.scoring_engine.calculate_fundamental_score(
            inputs.fundamental, stock.sector
        )
        sentiment_score = self.scoring_engine.calculate_sentiment_score(inputs.sentiment)

        # 板块/相对强度：用个股 20 日动量相对宽基指数动量的真实相对强度（RS）。
        # 指数动量来自 _detect_market_status（P2，真实 K 线）。
        from data_providers import relative_strength_score
        index_mom20 = self._market_status.get("trend_strength", 0.0) if self._market_status else 0.0
        sector_score = relative_strength_score(
            float(inputs.technical.get("momentum_20d", 0)), index_mom20
        )

        # 市场状态感知的动态权重
        weights = ConfigManager.get_market_adjusted_weights(self._market_regime)

        # 收盘预测特殊处理
        if analysis_type == 'evening':
            from evening_optimizer import EveningPredictionOptimizer
            evening_optimizer = EveningPredictionOptimizer()
            final_score, signal, reasons = evening_optimizer.optimize_evening_prediction(
                technical_score, fundamental_score, sentiment_score, sector_score,
                datetime.now(), weights=weights
            )
            confidence = 65  # 收盘预测使用固定信心值
        else:
            # 其他时间段的正常预测
            final_score = self.scoring_engine.calculate_final_score(
                technical_score, fundamental_score, sentiment_score, sector_score, weights=weights
            )
            signal, confidence, reasons = self.signal_generator.generate_signal(
                final_score, stock, inputs.technical
            )
        
        return PredictionResult(
            stock=stock,
            current_price=inputs.current_price,
            change_percent=inputs.change_percent,
            technical_score=technical_score,
            fundamental_score=fundamental_score,
            sentiment_score=sentiment_score,
            sector_score=sector_score,
            final_score=final_score,
            signal=signal,
            confidence=confidence,
            reasons=reasons,
            prediction_time=datetime.now(),
            data_provenance=inputs.provenance,
        )
    
    def predict_portfolio(self, stocks: List[StockConfig], analysis_type: str = "evening") -> List[PredictionResult]:
        """对股票组合进行预测：逐只先算公式，再叠加 LLM（若可用）。

        逐只调用而非批量：免费模型下批量请求必然在 60s 超时并退化成逐只，
        既没省调用还白等 60s；逐只路径已验证稳定（5/5），且能天然隔离
        单只行情/解析失败（单股异常被 try/except 跳过，不影响其余）。"""
        
        self._detect_market_status()
        self._skipped_stocks = []

        # ── 逐只预测（每只独立尝试 LLM 叠加，失败则公式）──
        results = []
        if os.environ.get("STOCK_SKIP_AGENT") == "1":
            # cron 路径：纯公式无 LLM 调用，线程安全 → 并行拉数，
            # 把全程压进 300s 内（agentTurn 轮询耐心上限，2026-09-14 事故）
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=8) as ex:
                futs = {ex.submit(self.predict_stock, s, analysis_type): s
                        for s in stocks}
                for fut, s in futs.items():
                    try:
                        results.append(fut.result())
                    except Exception as e:
                        reason = f"{s.name}({s.symbol}): 行情源瞬时失败未纳入"
                        print(f"[WARN] 跳过 {s.name}（分析失败: {e}），不参与本次分析")
                        self._skipped_stocks.append(reason)
        else:
            for stock in stocks:
                try:
                    result = self.predict_stock(stock, analysis_type)
                except Exception as e:
                    reason = f"{stock.name}({stock.symbol}): 行情源瞬时失败未纳入"
                    print(f"[WARN] 跳过 {stock.name}（分析失败: {e}），不参与本次分析")
                    self._skipped_stocks.append(reason)
                    continue
                results.append(result)

        if not results:
            print("[WARN] 全部股票分析失败，无可用预测")
            return []

        # ── 低置信降权（股神风控：防止近期方向多次反向的标的被误推买入）──
        self._apply_low_confidence(results)

        # ── 路线 A 大盘趋势过滤（股神 8-24 批准）：下跌趋势时买入→持有 ──
        self._apply_market_trend_gate(results)

        results.sort(key=lambda x: x.final_score, reverse=True)
        return results

    def _apply_low_confidence(self, results: List[PredictionResult]) -> None:
        """低置信降权：仅拦截原本会进入买入推荐的标的（final_score>=买入阈值），
        将其压到买入区以下；本就不买入的标的保持原信号，避免制造卖出误信号。
        降权幅度按当日方向不一致率自适应，命中率回升则自动解除（非永久剔除）。"""
        low_conf = _load_low_confidence_symbols()
        if not low_conf:
            return
        LOW_CONF_MAX_PENALTY = 1.5
        try:
            b_line = ConfigManager.get_signal_thresholds().get('buy', 5.8)
        except Exception:
            b_line = 5.8
        for r in results:
            mr = low_conf.get(r.stock.symbol)
            if mr is None:
                continue
            # 本就不会被买入推荐：不降权，保留原信号（不制造卖出误信号）
            if r.final_score < b_line:
                continue
            penalty = round(min(mr, 1.0) * LOW_CONF_MAX_PENALTY, 1)
            if penalty <= 0:
                continue
            old_score = r.final_score
            new_score = round(r.final_score - penalty, 2)
            if new_score >= b_line:
                new_score = round(b_line - 0.1, 2)
            r.final_score = max(1.0, new_score)
            r.reasons.append(f"低置信·近期方向多次不一致(买入拦截, 降至{r.final_score})")
            # 仅重算 signal 字符串保持与分数一致，不覆盖 reasons/confidence
            r.signal = self.signal_generator.generate_signal(r.final_score, r.stock, None)[0]
            print(f"[低置信买入拦截] {r.stock.name}({r.stock.symbol}) "
                  f"{old_score}→{r.final_score} (miss_rate={mr})")


# ==================== 评分引擎 ====================

class ScoringEngine:
    """评分引擎 - 计算各项评分（分段线性，减少阶梯跳变）"""

    @staticmethod
    def _lin(x: float, knots: List[Tuple[float, float]]) -> float:
        k = sorted(knots, key=lambda t: t[0])
        if x <= k[0][0]:
            return float(k[0][1])
        for i in range(1, len(k)):
            x0, y0 = k[i - 1]
            x1, y1 = k[i]
            if x <= x1:
                den = x1 - x0
                t = 0.0 if den == 0 else (x - x0) / den
                return float(y0 + t * (y1 - y0))
        return float(k[-1][1])
    
    def calculate_technical_score(self, data: Dict) -> float:
        """计算技术面评分（基于真实 K 线多周期指标）"""
        rsi = float(data.get("rsi", 50))
        rsi_score = self._lin(
            rsi,
            [(0, 1), (25, 9), (35, 7), (45, 5), (55, 4), (65, 3), (75, 2), (100, 1)],
        )

        sig = str(data.get("macd_signal", "中性"))
        mp = -1.0 if sig == "死叉" else (1.0 if sig == "金叉" else 0.0)
        macd_score = self._lin(mp, [(-1, 2.2), (0, 5.0), (1, 8.2)])

        bollinger = float(data.get("bollinger_position", 0.5))
        bollinger_score = self._lin(
            bollinger, [(0, 9), (0.2, 7.5), (0.4, 6), (0.5, 5), (0.6, 4), (0.8, 3), (1, 1)]
        )

        volume = float(data.get("volume_ratio", 1))
        volume_score = self._lin(
            volume, [(0.3, 2), (0.8, 4), (1.1, 6), (1.4, 8), (2.5, 8.5)]
        )

        momentum = float(data.get("momentum_5d", 0))
        momentum_score = self._lin(
            momentum, [(-0.05, 2), (-0.02, 4), (0, 6), (0.02, 6.5), (0.05, 8)]
        )

        # 多周期趋势强度（新增）：均线排列 + 20日动量
        alignment = str(data.get("ma_alignment", "中性"))
        align_score = {"多头排列": 8.5, "均线缠绕": 5.0, "空头排列": 2.0, "中性": 5.0}.get(alignment, 5.0)

        mom20 = float(data.get("momentum_20d", 0))
        trend_score = self._lin(
            mom20, [(-0.15, 2), (-0.05, 4), (0, 5.5), (0.05, 7), (0.15, 8.5)]
        )

        # 加权：5 个核心指标 + 2 个趋势指标
        core = (rsi_score + macd_score + bollinger_score + volume_score + momentum_score) / 5
        trend = (align_score + trend_score) / 2
        final = core * 0.7 + trend * 0.3
        return round(final, 1)
    
    def calculate_fundamental_score(self, data: Dict, sector: str) -> float:
        """计算基本面评分"""
        
        benchmark = ConfigManager.get_sector_benchmark(sector)
        pe_avg = float(benchmark["pe_avg"]) or 1.0
        pb_avg = float(benchmark["pb_avg"]) or 1.0
        roe_avg = float(benchmark["roe_avg"]) or 1.0
        growth_avg = float(benchmark["growth_avg"]) or 1.0

        r_pe = float(data.get("pe_ratio", pe_avg)) / pe_avg
        pe_score = self._lin(r_pe, [(0.4, 9), (0.75, 7), (1.0, 5.5), (1.25, 4), (1.6, 2)])

        r_pb = float(data.get("pb_ratio", pb_avg)) / pb_avg
        pb_score = self._lin(r_pb, [(0.4, 9), (0.75, 7), (1.0, 5.5), (1.25, 4), (1.6, 2)])

        r_roe = float(data.get("roe", roe_avg)) / roe_avg
        roe_score = self._lin(r_roe, [(0.5, 3), (0.85, 5), (1.0, 6.5), (1.2, 8), (1.5, 9)])

        r_gr = float(data.get("growth_rate", growth_avg)) / growth_avg
        growth_score = self._lin(r_gr, [(0.5, 3), (0.85, 5), (1.0, 6.5), (1.2, 8), (1.5, 9)])

        debt = float(data.get("debt_ratio", 0.5))
        debt_score = self._lin(debt, [(0.15, 9), (0.35, 7), (0.5, 5.5), (0.65, 4), (0.85, 2)])

        divy = float(data.get("dividend_yield", 0))
        dividend_score = self._lin(divy, [(0, 2), (1, 4), (2, 6), (3, 8), (5, 9)])
        
        return round((pe_score + pb_score + roe_score + growth_score + debt_score + dividend_score) / 6, 1)
    
    def calculate_sentiment_score(self, data: Dict) -> float:
        """计算情绪面评分"""
        
        heat_score = float(data.get("market_heat", 5))
        attention_score = float(data.get("institution_attention", 5))

        retail_map = {"恐慌": 2.0, "谨慎": 4.0, "中性": 6.0, "乐观": 8.0, "狂热": 4.5}
        retail_score = retail_map.get(str(data.get("retail_sentiment", "中性")), 6.0)

        news_map = {"负面": 3.0, "中性": 6.0, "正面": 8.0}
        news_score = news_map.get(str(data.get("news_sentiment", "中性")), 6.0)
        
        return round((heat_score + attention_score + retail_score + news_score) / 4, 1)
    
    def calculate_final_score(self, technical: float, fundamental: float, sentiment: float, sector: float, weights: Optional[Dict[str, float]] = None) -> float:
        """计算综合评分（weights 为空时用基准权重）"""
        
        if weights is None:
            weights = ConfigManager.get_score_weights()
        final_score = (
            technical * weights['technical'] +
            fundamental * weights['fundamental'] +
            sentiment * weights['sentiment'] +
            sector * weights['sector']
        )
        return round(final_score, 1)


# ==================== 信号生成器 ====================

def _confidence_from_score_and_technical(final_score: float, technical: Dict) -> int:
    """由综合分与 RSI 偏离度推导信心（可复现，非随机）。"""
    th = ConfigManager.get_signal_thresholds()
    dist_to_edges = [
        abs(final_score - th["strong_buy"]),
        abs(final_score - th["buy"]),
        abs(final_score - th["hold"]),
        abs(final_score - th["sell"]),
    ]
    margin = min(dist_to_edges)
    base = int(42 + final_score * 4.5)
    rsi = float(technical.get("rsi", 50))
    rsi_bump = int(min(10, abs(rsi - 50) / 5.0))
    margin_bump = int(min(10, margin * 2.5))
    return max(40, min(91, base + (rsi_bump + margin_bump) // 2))


class SignalGenerator:
    """信号生成器 - 生成交易信号和理由"""
    
    def generate_signal(
        self, final_score: float, stock: StockConfig, technical_data: Optional[Dict] = None
    ) -> Tuple[str, int, List[str]]:
        """生成交易信号"""
        
        technical_data = technical_data or {}
        thresholds = ConfigManager.get_signal_thresholds()
        
        if final_score >= thresholds['strong_buy']:
            signal = SignalType.STRONG_BUY
        elif final_score >= thresholds['buy']:
            signal = SignalType.BUY
        elif final_score >= thresholds['hold']:
            signal = SignalType.HOLD
        elif final_score >= thresholds['sell']:
            signal = SignalType.SELL
        else:
            signal = SignalType.STRONG_SELL
        
        sig_str = signal.value
        margin = float(ConfigManager.get_accuracy_tuning().get("signal_margin") or 0)
        sig_str = apply_signal_margin(sig_str, final_score, thresholds, margin)

        confidence = _confidence_from_score_and_technical(final_score, technical_data)
        reasons = self._generate_reasons(final_score, stock)
        
        return sig_str, confidence, reasons
    
    def _generate_reasons(self, final_score: float, stock: StockConfig) -> List[str]:
        """生成推荐理由"""
        
        reasons = []
        
        # 基础理由
        if final_score >= 8:
            reasons.extend(["技术面强势突破", "基本面优质低估"])
        elif final_score >= 6:
            reasons.extend(["技术面有所改善", "估值相对合理"])
        elif final_score >= 4:
            reasons.extend(["处于震荡整理阶段", "等待明确方向"])
        else:
            reasons.extend(["技术面走弱", "存在调整压力"])
        
        # 行业特定理由
        sector_reasons = {
            '白酒': ["消费复苏预期", "品牌溢价能力强"],
            '新能源': ["政策支持持续", "长期成长空间大"],
            '银行': ["息差改善预期", "资产质量稳定"],
            '医药': ["刚需属性突出", "创新药进展"],
            '科技': ["技术创新驱动", "国产替代逻辑"],
            '消费': ["消费升级趋势", "渠道优势明显"],
            '地产': ["政策边际改善", "估值处于低位"],
            '面板': ["供需格局改善", "价格上涨预期"]
        }
        
        if stock.sector in sector_reasons:
            opts = sector_reasons[stock.sector]
            pick = opts[abs(hash(stock.symbol)) % len(opts)]
            reasons.append(pick)
        
        return reasons[:2]  # 限制为2个理由


# ==================== 总结引擎 ====================

class SummaryEngine:
    """总结引擎 - 基于预测结果生成总结报告"""
    
    def generate_summary(self, predictions: List[PredictionResult], analysis_type: str, market_status: Optional[Dict] = None) -> SummaryReport:
        """生成总结报告"""
        th = ConfigManager.get_signal_thresholds()
        b_line, h_line = th["buy"], th["hold"]
        # 分类推荐（与 SignalGenerator 档位一致）
        # 买卖信号（方向性）全部展示——它们是自校准的方向性样本来源，不能截断；
        # 持有信号数量多，封顶展示 8 只保持报告可读。
        buy_recommendations = [p for p in predictions if p.final_score >= b_line]
        sell_recommendations = [p for p in predictions if p.final_score < h_line]
        hold_recommendations = [p for p in predictions if h_line <= p.final_score < b_line][:8]
        
        # 市场概况
        market_overview = self._generate_market_overview(predictions, buy_recommendations, sell_recommendations, hold_recommendations, market_status)
        
        # 行业分析
        sector_analysis = self._generate_sector_analysis(predictions)
        
        # 风险提示
        risk_alerts = self._generate_risk_alerts(predictions, market_status)
        
        # 下一步行动
        next_actions = self._generate_next_actions(analysis_type, predictions)
        
        return SummaryReport(
            report_time=datetime.now(),
            analysis_type=analysis_type,
            buy_recommendations=buy_recommendations,
            sell_recommendations=sell_recommendations,
            hold_recommendations=hold_recommendations,
            market_overview=market_overview,
            sector_analysis=sector_analysis,
            risk_alerts=risk_alerts,
            next_actions=next_actions
        )
    
    def _generate_market_overview(self, predictions: List[PredictionResult], buy: List[PredictionResult], 
                                sell: List[PredictionResult], hold: List[PredictionResult], 
                                market_status: Optional[Dict] = None) -> Dict:
        """生成市场概况"""
        
        total_buy = len(buy)
        total_sell = len(sell)
        total_hold = len(hold)
        
        avg_buy_score = sum(p.final_score for p in buy) / total_buy if total_buy > 0 else 0
        avg_sell_score = sum(p.final_score for p in sell) / total_sell if total_sell > 0 else 0
        avg_hold_score = sum(p.final_score for p in hold) / total_hold if total_hold > 0 else 0
        
        buy_sectors = list(set(p.stock.sector for p in buy))
        sell_sectors = list(set(p.stock.sector for p in sell))
        hold_sectors = list(set(p.stock.sector for p in hold))
        
        overview = {
            'total_stocks': len(predictions),
            'buy_count': total_buy,
            'sell_count': total_sell,
            'hold_count': total_hold,
            'avg_buy_score': round(avg_buy_score, 1),
            'avg_sell_score': round(avg_sell_score, 1),
            'avg_hold_score': round(avg_hold_score, 1),
            'buy_sectors': buy_sectors,
            'sell_sectors': sell_sectors,
            'hold_sectors': hold_sectors,
            'market_sentiment': '乐观' if avg_buy_score > avg_sell_score else '谨慎' if avg_sell_score > avg_buy_score else '中性'
        }
        # 叠加市场状态（regime 感知）
        if market_status:
            overview['market_status'] = market_status.get('status', 'ranging')
            overview['market_regime'] = market_status.get('regime', 'range')
            overview['market_volatility'] = market_status.get('volatility', 0.0)
            overview['market_trend_strength'] = market_status.get('trend_strength', 0.0)
            overview['market_status_desc'] = market_status.get('description', '')
            if market_status.get('index_trend_gate') == 'down_trend':
                overview['index_trend_gate'] = '下跌趋势·买入信号已暂停(仅持有)'
        return overview
    
    def _generate_sector_analysis(self, predictions: List[PredictionResult]) -> Dict:
        """生成行业分析"""
        
        sector_scores = {}
        for pred in predictions:
            sector = pred.stock.sector
            if sector not in sector_scores:
                sector_scores[sector] = []
            sector_scores[sector].append(pred.final_score)
        
        sector_analysis = {}
        for sector, scores in sector_scores.items():
            avg_score = sum(scores) / len(scores)
            sector_analysis[sector] = {
                'avg_score': round(avg_score, 1),
                'stock_count': len(scores),
                'trend': '强势' if avg_score > 6 else '弱势' if avg_score < 4 else '震荡'
            }
        
        return sector_analysis
    
    def _generate_risk_alerts(self, predictions: List[PredictionResult], market_status: Optional[Dict] = None) -> List[str]:
        """生成风险提示"""
        
        alerts = []
        
        # 路线 A 大盘闸门：下跌趋势时主动提示（买入已暂停）
        if market_status and market_status.get("index_trend_gate") == "down_trend":
            alerts.append("大盘处于下跌趋势（上证20/60DMA空头），买入信号已暂停，仅作持有观察")
        
        # 检查是否有大量卖出信号
        sell_count = len([p for p in predictions if p.final_score < 5])
        if sell_count > len(predictions) * 0.4:
            alerts.append("市场卖出信号较多，注意风险控制")
        
        # 检查是否有极端分数
        extreme_scores = [p for p in predictions if p.final_score < 2 or p.final_score > 9]
        if extreme_scores:
            alerts.append(f"发现{len(extreme_scores)}只股票评分极端，需谨慎对待")
        
        # 检查行业风险
        sector_analysis = self._generate_sector_analysis(predictions)
        weak_sectors = [s for s, data in sector_analysis.items() if data['avg_score'] < 4]
        if weak_sectors:
            alerts.append(f"弱势行业: {', '.join(weak_sectors)}")
        
        return alerts[:3]  # 限制为3个提示
    
    def _generate_next_actions(self, analysis_type: str, predictions: List[PredictionResult]) -> List[str]:
        """生成下一步行动建议"""
        
        actions = []
        
        if analysis_type == 'morning':
            actions.append("关注开盘后的量价配合情况")
            actions.append("观察昨夜美股对A股的影响")
            actions.append("留意早盘资金流入方向")
        elif analysis_type == 'afternoon':
            actions.append("关注午后资金动向")
            actions.append("观察上午强势股的持续性")
            actions.append("留意尾盘异动情况")
        elif analysis_type == 'evening':
            actions.append("总结全天市场表现")
            actions.append("为次日交易做准备")
            actions.append("关注晚间重要消息")
        elif analysis_type == 'weekly':
            actions.append("回顾本周交易策略执行情况")
            actions.append("制定下周投资计划")
            actions.append("关注周末政策消息")
        
        return actions


# ==================== 配置管理器 ====================

class ConfigManager:
    """配置管理器"""
    
    _core_stocks_cache: Optional[List[StockConfig]] = None
    _analysis_limits_cache: Optional[Dict[str, int]] = None
    
    _DEFAULT_CORE_STOCKS = [
        StockConfig('贵州茅台', '600519', '白酒', 0.9),
        StockConfig('宁德时代', '300750', '新能源', 0.8),
        StockConfig('招商银行', '600036', '银行', 0.7),
        StockConfig('五粮液', '000858', '白酒', 0.6),
        StockConfig('恒瑞医药', '600276', '医药', 0.5),
        StockConfig('比亚迪', '002594', '新能源', 0.4),
        StockConfig('海康威视', '002415', '科技', 0.3),
        StockConfig('伊利股份', '600887', '消费', 0.2),
        StockConfig('万科A', '000002', '地产', 0.1),
        StockConfig('京东方A', '000725', '面板', 0.0)
    ]
    
    # 兼容旧代码引用
    CORE_STOCKS = _DEFAULT_CORE_STOCKS
    
    @classmethod
    def _stock_pool_path(cls) -> Path:
        root = Path(_default_stock_system_root())
        return root / "config" / "stock_pool.json"
    
    @classmethod
    def reload_stock_pool(cls) -> None:
        """从 config/stock_pool.json 重新加载；文件不存在则用内置默认。"""
        cls._core_stocks_cache = None
        cls._analysis_limits_cache = None
    
    @classmethod
    def get_core_stocks(cls) -> List[StockConfig]:
        if cls._core_stocks_cache is not None:
            return cls._core_stocks_cache
        path = cls._stock_pool_path()
        if not path.is_file():
            cls._core_stocks_cache = list(cls._DEFAULT_CORE_STOCKS)
            cls._analysis_limits_cache = {"morning": 20, "default": 20}
            return cls._core_stocks_cache
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            stocks_raw = data.get("stocks") or []
            out: List[StockConfig] = []
            for i, row in enumerate(stocks_raw):
                out.append(
                    StockConfig(
                        name=row["name"],
                        symbol=row["symbol"],
                        sector=row["sector"],
                        weight=float(row.get("weight", 1.0 - i * 0.1)),
                    )
                )
            cls._core_stocks_cache = out if out else list(cls._DEFAULT_CORE_STOCKS)
            lim = data.get("analysis_limits") or {}
            cls._analysis_limits_cache = {
                "morning": int(lim.get("morning", 5)),
                "afternoon": int(lim.get("afternoon", lim.get("default", 10))),
                "evening": int(lim.get("evening", lim.get("default", 10))),
                "weekly": int(lim.get("weekly", lim.get("default", 10))),
                "default": int(lim.get("default", 10)),
            }
        except (OSError, ValueError, KeyError, TypeError):
            cls._core_stocks_cache = list(cls._DEFAULT_CORE_STOCKS)
            cls._analysis_limits_cache = {"morning": 20, "default": 20}
        return cls._core_stocks_cache
    
    @classmethod
    def get_analysis_stock_slice(cls, analysis_type: str) -> List[StockConfig]:
        stocks = cls.get_core_stocks()
        lims = cls._analysis_limits_cache or {"morning": 20, "default": 20}
        n = lims.get(analysis_type, lims.get("default", len(stocks)))
        n = max(1, min(n, len(stocks)))
        # 固定自选股池：每天跑同一批（由 config/stock_pool.json 的 stocks 决定），
        # 不再按交易日轮换抽样。这样能持续跟踪同一批个股、稳定积累方向性样本，
        # 供 auto_calibration 跨日学习（否则买卖信号样本长期稀缺、校准空转）。
        return stocks[:n]

    # 评分权重配置
    SCORE_WEIGHTS = {
        "technical": 0.40,
        "fundamental": 0.35,
        "sentiment": 0.15,
        "sector": 0.10,
    }

    # 市场状态 → 动态权重（乘性微调，量级 ±0.1，不颠覆基准）
    #  trend      : 趋势市顺势而为，技术面占优
    #  range      : 震荡市均值回归，基本面/估值占优
    #  volatility : 高波动市风控优先，情绪面占优
    MARKET_REGIME_WEIGHTS = {
        "trend":      {"technical": 0.50, "fundamental": 0.28, "sentiment": 0.13, "sector": 0.09},
        "range":      {"technical": 0.30, "fundamental": 0.45, "sentiment": 0.13, "sector": 0.12},
        "volatility": {"technical": 0.28, "fundamental": 0.30, "sentiment": 0.30, "sector": 0.12},
    }

    # 信号阈值配置（默认值；运行时可由 config/calibration_overrides.json 覆盖）
    # 2026-08-11 调窄：真实综合分分布约 4.58~6.54（中位 5.70），旧阈值 buy≥7.0 几乎永远触发不了买入。
    # 新阈值让“略高于中位→买入、略低于 5.0→卖出”，使系统能产生方向性信号以积累校准样本。
    SIGNAL_THRESHOLDS = {
        "strong_buy": 6.5,
        "buy": 5.8,
        "hold": 5.0,
        "sell": 4.3,
    }

    _merged_signal_thresholds: Optional[Dict[str, float]] = None
    _merged_score_weights: Optional[Dict[str, float]] = None
    _merged_accuracy_tuning: Optional[Dict[str, Any]] = None

    @classmethod
    def reload_calibration(cls) -> None:
        """清除合并缓存，下次读取磁盘上的 calibration_overrides.json。"""
        cls._merged_signal_thresholds = None
        cls._merged_score_weights = None
        cls._merged_accuracy_tuning = None

    @classmethod
    def _merge_from_calibration(cls, key: str, base: Dict, *, normalize: bool = False) -> Dict:
        """通用合并：从 calibration_overrides.json 覆盖 base 中同名键。"""
        path = Path(_default_stock_system_root()) / "config" / "calibration_overrides.json"
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                ov = data.get(key) or {}
                for k in base:
                    if k in ov:
                        base[k] = float(ov[k])
                if normalize:
                    s = sum(base.values())
                    if s > 0:
                        base = {k: round(base[k] / s, 4) for k in base}
            except (OSError, ValueError, TypeError, KeyError):
                pass
        return base

    @classmethod
    def get_accuracy_tuning(cls) -> Dict[str, Any]:
        if cls._merged_accuracy_tuning is not None:
            return cls._merged_accuracy_tuning
        base = default_accuracy_tuning()
        path = Path(_default_stock_system_root()) / "config" / "calibration_overrides.json"
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                ov = data.get("accuracy_tuning") or {}
                base = merge_accuracy_tuning(ov)
            except (OSError, ValueError, TypeError, KeyError):
                pass
        cls._merged_accuracy_tuning = base
        return base

    @classmethod
    def get_signal_thresholds(cls) -> Dict[str, float]:
        if cls._merged_signal_thresholds is not None:
            return cls._merged_signal_thresholds
        cls._merged_signal_thresholds = cls._merge_from_calibration(
            "signal_thresholds", dict(cls.SIGNAL_THRESHOLDS)
        )
        return cls._merged_signal_thresholds

    @classmethod
    def get_score_weights(cls) -> Dict[str, float]:
        if cls._merged_score_weights is not None:
            return cls._merged_score_weights
        cls._merged_score_weights = cls._merge_from_calibration(
            "score_weights", dict(cls.SCORE_WEIGHTS), normalize=True
        )
        return cls._merged_score_weights

    @classmethod
    def get_market_adjusted_weights(cls, regime: str) -> Dict[str, float]:
        """根据市场状态返回动态权重（可被 calibration_overrides.json 覆盖）。

        regime ∈ {trend, range, volatility}；未知值回落到基准 SCORE_WEIGHTS。
        """
        base = dict(cls.MARKET_REGIME_WEIGHTS.get(regime, cls.SCORE_WEIGHTS))
        return cls._merge_from_calibration(
            f"regime_weights_{regime}", base, normalize=True
        )
    
    @classmethod
    def get_sector_benchmark(cls, sector: str) -> Dict[str, float]:
        """获取行业基准"""
        return SECTOR_BENCHMARKS.get(sector, {
            'pe_avg': 20, 'pb_avg': 3, 'roe_avg': 15, 'growth_avg': 15
        })


# ==================== 报告生成器 ====================

class ReportGenerator:
    """报告生成器 - 生成文本报告"""
    
    def __init__(self, base_dir: Optional[str] = None):
        root = base_dir or _default_stock_system_root()
        self.base_dir = Path(root)
        self.data_dir = self.base_dir / "data"
        self.reports_dir = self.base_dir / "reports"
        self.logs_dir = self.base_dir / "logs"
        
        # 确保目录存在
        self.data_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

    def _portfolio_action_lines(self) -> List[str]:
        """【今日操作指令】—— 系统下达明确的买卖指令，不把判断推回给用户。

        风险偏好是**一个输入参数**（可承受最大回撤，见 data/portfolio.json 的
        max_drawdown_tolerance），系统据此自动反推股债比例并生成指令。
        用户要做的只是照着执行；没有歧义、不做主观判断。
        """
        try:
            here = str(Path(__file__).resolve().parent)
            if here not in sys.path:
                sys.path.insert(0, here)
            import portfolio_manager as pm
            pf = pm.load()
            if not pf or not pf.get("holdings"):
                pf = pm.init()          # 首次自动建立虚拟跟踪持仓
            acts = pm.actions(pf)
        except Exception:
            return []        # 取价失败/未初始化 → 静默，不播报错误信息
        if not acts:
            return []
        out = ["", "【今日操作指令 · 按系统执行】", "-" * 60]
        if pf.get("virtual"):
            out.append("（当前为虚拟跟踪持仓；改实际持仓请编辑 "
                       "data/portfolio.json 的 holdings/cash）")
        for a in acts:
            out.append("  " + a)
        out.append(f"  规则：偏离目标 ≥5pp 触发再平衡；中性目标来自可承受回撤 "
                   f"{pf.get('max_drawdown_tolerance', 0):.0%} "
                   f"→ 股票 {pf.get('target_stock_ratio', 0):.0%}")
        try:
            _root = Path(__file__).resolve().parent.parent
            _tv = json.loads((_root / "data" / "timing_tilt_validate.json")
                             .read_text(encoding="utf-8"))
            _t = _tv.get("excess_t_nw", _tv.get("excess_t", 0))
            _tp = _tv.get("sub_post", {}).get("t", 0)
        except Exception:
            _t, _tp = 0.0, 0.0
        out.append("  估值倾斜已启用：沪深300 便宜(低于5年线)自动加股比、"
                   f"贵时自动降股比（扣基准 t_NW={_t:+.2f}，但2015后 t={_tp:+.2f}"
                   "偏弱未过门槛，主要价值=控回撤，非圣杯）")
        out.append("-" * 60)
        return out

    def _track_lines(self) -> List[str]:
        """【实盘跟踪】—— 按系统指令执行的组合实际收益率，每日更新。

        这是检验系统价值的真值：跟踪盘收益 vs 60/40 买入持有 vs 满仓股。
        报告生成时若当日净值未记录，自动触发 track_performance.py 盯市一次
        （早报每日必跑 → 净值必然每日记录），再读取 portfolio_track_report.json。
        """
        out = []
        try:
            root = Path(__file__).resolve().parent.parent
            rpt = root / "data" / "portfolio_track_report.json"
            # 当日净值保障：报告缺失 / last_date落后今天 / 陈旧(>3日) → 现场盯市
            import subprocess, time as _time
            stale = True
            try:
                _tv0 = json.loads(rpt.read_text(encoding="utf-8"))
                _today = _time.strftime("%Y-%m-%d")
                _age = _time.time() - rpt.stat().st_mtime
                stale = (_tv0.get("last_date") != _today) or _age > 3 * 86400
            except Exception:
                stale = True
            if stale:
                try:
                    subprocess.run(
                        [sys.executable, str(root / "refactored" / "track_performance.py")],
                        capture_output=True, timeout=60)
                except Exception:
                    pass
            tv = json.loads(rpt.read_text(encoding="utf-8"))
        except Exception:
            return []
        out += ["", "【实盘跟踪 · 按系统执行】", "-" * 60]
        out.append(f"  跟踪盘（虚拟{tv.get('initial_total',0):,.0f}元，"
                   f"{tv.get('base_date','')} 建仓，已跟踪 {tv.get('days_tracked',0)} 日）："
                   f"累计 {tv.get('track_ret_pct',0):+.2f}%"
                   f"（现值 {tv.get('total_now',0):,.0f} 元）")
        out.append(f"  同期基准: 沪深300ETF {tv.get('bench_300_ret_pct',0):+.2f}% / "
                   f"60/40不操作 {tv.get('bench_6040_ret_pct',0):+.2f}%")
        out.append(f"  系统操作超额(vs 60/40): {tv.get('excess_vs_6040_pp',0):+.2f}pp"
                   f" ← 每日更新，检验系统是否值得")
        if tv.get("short_sample"):
            out.append("  （跟踪不足60日，数字无统计意义，机制演示期）")
        out.append("-" * 60)
        return out

    def _allocation_lines(self) -> List[str]:
        """【配置指导】—— 系统唯一有历史验证支持的买卖框架。

        不依赖任何预测：股债配置 + 定期再平衡。再平衡天然产生明确的
        买卖指令（股票涨超目标→卖股买债；跌破→卖债买股）。
        所有数字动态读 data/asset_allocation_report.json，禁止硬编码。
        """
        try:
            root = Path(__file__).resolve().parent.parent
            rep = json.loads((root / "data" / "asset_allocation_report.json")
                             .read_text(encoding="utf-8"))
        except Exception:
            return []
        rows = {r["config"]: r for r in rep.get("rows", [])}
        if not rows:
            return []
        out = ["", "【配置指导 · 唯一有历史验证的买卖框架】", "-" * 60]
        out.append(f"标的: 股票 {rep.get('stock', '')} / 债券 {rep.get('bond', '')}")
        out.append("目标配比: 股票60% + 债券40%（求稳可用50/50）")
        out.append("调仓规则: 每年固定日再平衡回目标 —— 这是明确的买卖指令，"
                   "不依赖任何预测")
        out.append(f"历史验证 ({rep.get('start', '')}~{rep.get('end', '')}):")
        for key in ("100% 股票(基准)", "60/40 年度再平衡", "50/50 年度再平衡"):
            r = rows.get(key)
            if r:
                out.append(f"  {key:<18} 年化{r['cagr']:>+6.2f}%  "
                           f"回撤{r['mdd']:>6.1f}%  夏普{r['sharpe']:>4.2f}  "
                           f"{r['n_trades']}次调仓")
        base = rows.get("100% 股票(基准)")
        best_rb = rows.get("60/40 年度再平衡")
        if base and best_rb:
            dcagr = best_rb["cagr"] - base["cagr"]
            dsh = best_rb["sharpe"] - base["sharpe"]
            out.append(f"→ 与满仓股票相比: 收益{dcagr:+.2f}%/年（基本持平），"
                       f"回撤从{base['mdd']:.0f}%降到{best_rb['mdd']:.0f}%，"
                       f"夏普{base['sharpe']:.2f}→{best_rb['sharpe']:.2f}"
                       f"（{dsh:+.2f}）。")
            out.append("→ 配债不是为了多赚，是为了**拿得住**："
                       "满仓股要扛-70%回撤不割肉才有它的长期收益。")
        # 调仓频率：数据证明"买卖点不是越频繁越好"，年度最优
        try:
            rf = json.loads((root / "data" / "rebalance_freq_report.json")
                            .read_text(encoding="utf-8"))
            rr = {r["freq"]: r for r in rf.get("rows", [])}
            pick = [k for k in ("日频(每天)", "月频(21日)", "年度(244日)")
                    if k in rr]
            if len(pick) == 3:
                out.append("买卖点频率（23.6年实测，为什么要一年只动一次）:")
                for k in pick:
                    out.append(f"  {k:<12} 年化{rr[k]['cagr']:+.2f}% "
                               f"回撤{rr[k]['mdd']:.0f}% "
                               f"（{rr[k]['n_rebal']}次）")
                out.append("  → 频率越高越差（成本+噪音），太稀疏也差"
                           "（偏离太久回撤变大）；年度是唯一最优点。")
        except Exception:
            pass

        # 持有期 → 亏损概率（回撤只能用时间消除，不能用策略消除）
        try:
            hp = json.loads((root / "data" / "hold_period_report.json")
                            .read_text(encoding="utf-8"))
            v8 = hp.get("8", {}).get("60/40再平衡")
            v1 = hp.get("1", {}).get("60/40再平衡")
            v10 = hp.get("10", {}).get("满仓股票")
            if v8:
                out.append("持有期→亏损概率（23.6年滚动窗口实测）:")
                if v1:
                    out.append(f"  持1年: 亏损概率{v1['loss_prob']:.0%}，"
                               f"最差{v1['worst']:+.0f}%")
                out.append(f"  持8年: 亏损概率{v8['loss_prob']:.0%}，"
                           f"最差{v8['worst']:+.0f}%，中位{v8['median']:+.0f}%")
                if v10:
                    out.append(f"  （对比满仓股持10年仍有{v10['loss_prob']:.0%}"
                               f"概率亏，最差{v10['worst']:+.0f}%）")
                out.append("  → 回撤无法用策略消除，只能用**持有时间**消除。")
        except Exception:
            pass
        # 估值倾斜（扣基准显著 t_NW=2.73 但2015后偏弱；双轨定位：回撤控制上生产/收益增强不宣传）
        try:
            tv = json.loads((root / "data" / "timing_tilt_validate.json")
                            .read_text(encoding="utf-8"))
            tlt, fx = tv.get("tilt", {}), tv.get("fixed", {})
            sp = tv.get("sub_post", {})
            t_nw = tv.get("excess_t_nw", tv.get("excess_t", 0))
            out.append("估值倾斜（便宜多买 / 贵时少买，回撤控制定位，已上生产）:")
            out.append(f"  机制: 沪深300 相对5年均线偏离 → 目标股比 60%±调"
                       f"（限40%~80%），年度再平衡")
            out.append(f"  回测: 倾斜 年化{tlt.get('cagr',0):+.2f}% 回撤"
                       f"{tlt.get('mdd',0):.0f}%  vs 固定60/40 "
                       f"{fx.get('cagr',0):+.2f}% / {fx.get('mdd',0):.0f}%")
            out.append(f"  诚实边界: 全样本扣基准 t_NW={t_nw:+.2f}"
                       f"（显著）；但2015后子样本 t_NW={sp.get('t',0):+.2f} 偏弱"
                       f"（未过四门槛③），近十年阴跌时失效")
            out.append("  → 已接入【今日操作指令】：贵时自动降股比、便宜自动加股比"
                       "，主要价值是控回撤，非多赚")
        except Exception:
            pass
        out.append("口径说明: 股票端已补年均股息2%（价格指数→全收益），"
                   "否则会系统性高估债券收益。")
        out.append("注意: 以上为被动配置规则，非预测信号；"
                   "个股/行业轮动信号均无实证支持，不据此交易。")
        out.append("-" * 60)
        return out

    def _trading_cost_lines(self) -> List[str]:
        """【交易成本账】—— 系统"不喊买卖"的量化价值。

        用实际预测日志算换手率 → 年化交易成本，与"不动"对比。
        全部动态计算：换手率读 rs_prediction_log.jsonl，成本按 A 股个股
        （佣金万2.5双边 + 卖出印花税0.05% + 滑点0.1%单边×2 ≈ 0.30%/来回）。
        核心信息：策略超额未达显著时，不动的期望优于按信号交易。
        """
        try:
            root = Path(__file__).resolve().parent.parent
            logf = root / "data" / "rs_prediction_log.jsonl"
            if not logf.exists():
                return []
            recs = []
            for ln in logf.read_text(encoding="utf-8").splitlines():
                try:
                    recs.append(json.loads(ln))
                except Exception:
                    pass
            recs.sort(key=lambda r: r.get("pred_date", ""))
        except Exception:
            return []
        turns = []
        for a, b in zip(recs, recs[1:]):
            sa, sb = set(a.get("top") or []), set(b.get("top") or [])
            if sa:
                turns.append(len(sa - sb) / len(sa))
        if len(turns) < 5:
            return []
        turn = sum(turns) / len(turns)
        round_trip = 0.0030          # A股个股一买一卖 ≈0.30%
        ann_cost = turn * round_trip * 252 * 100
        out = ["", "【交易成本账 · 为什么现在不建议按信号交易】", "-" * 60]
        out.append(f"当前预测日均换手 {turn:.0%}（{len(turns)} 日实测），"
                   f"按此交易年化成本 ≈ {ann_cost:.1f}%"
                   "（佣金+印花税+滑点，一买一卖约0.30%）")
        out.append("而预测超额尚未达显著——真实值可能是0甚至为负。")
        out.append("→ 不动的期望优于按信号交易；若需配置，等权持有宽基ETF"
                   "（21.6年实测年化+11.6%，成本≈0）更稳。")
        out.append("提示：多数散户亏损源于过度交易，而非选错标的。")
        out.append("-" * 60)
        return out

    def _rs_prediction_lines(self, top_n: int = 10) -> List[str]:
        """【相对强弱预测 TOP】—— 系统的核心预测输出（预测层，非描述层）。

        依据: 行业中性 60 日动量全池横截面排序（refactored/predict_relative_strength.py，
        walk-forward 157只×~700交易日）。置信状态动态读 data/rs_param_sweep.json，
        禁止硬编码统计数字。
        """
        try:
            root = Path(__file__).resolve().parent.parent
            rep = json.loads((root / "data" / "rs_prediction_report.json")
                             .read_text(encoding="utf-8"))
        except Exception:
            return []
        pred = rep.get("current_prediction") or []
        if not pred:
            return []

        # 动态置信声明 + 数据驱动的建议级别（计分卡核对历史预测准确性）
        conf = "置信状态: 未见参数扫描数据"
        advice_note = ""
        try:
            sweep = json.loads((root / "data" / "rs_param_sweep.json")
                               .read_text(encoding="utf-8")).get("summary", {})
            n = sweep.get("n_cells", 0)
            if n:
                pos = sweep.get("excess_positive", 0)
                sig = sweep.get("excess_significant", 0)
                lo, hi = sweep.get("excess_t_range", [0, 0])
                if sig > 0:
                    conf = (f"置信状态: 扣成本后 {sig}/{n} 组合显著为正 "
                            f"(t范围{lo:+.2f}~{hi:+.2f})")
                elif pos == n:
                    conf = (f"置信状态: 方向一致——{n}/{n} 组合超额为正但均未达显著 "
                            f"(t范围{lo:+.2f}~{hi:+.2f})")
                else:
                    conf = (f"置信状态: 未检出稳定 edge ({pos}/{n} 为正，"
                            f"t范围{lo:+.2f}~{hi:+.2f})")
        except Exception:
            pass
        try:
            sc = json.loads((root / "data" / "rs_prediction_scorecard.json")
                            .read_text(encoding="utf-8"))
            level = sc.get("advice_level", "OBSERVE")
            nc = sc.get("n_checked", 0)
            roll = sc.get("rolling", {})
            if level == "BUY_LEAN":
                advice_note = (f"🎯 建议级别: 买入倾向——滚动 {roll.get('n')} 次预测"
                               f"超额均值 {roll.get('mean', 0):+.3f}%/日 t={roll.get('t'):+.2f}"
                               f"命中 {roll.get('hit_rate', 0):.0%}，已过实证门槛，"
                               "TOP 列表可作买入候选")
            elif level == "AVOID":
                advice_note = (f"⚠️ 建议级别: 回避提示——滚动预测超额 "
                               f"{roll.get('mean', 0):+.3f}%/日 t={roll.get('t'):+.2f}"
                               f"显著为负，勿据 TOP 列表买入")
            else:
                if nc > 0:
                    advice_note = (f"建议级别: 观察——已核对 {nc} 次预测，"
                                   f"滚动超额 {roll.get('mean', 0):+.3f}%/日 "
                                   f"t={roll.get('t', 0):+.2f}，未达实证门槛"
                                   "（滚动 t 过 ±2 才升级/降级建议）")
                else:
                    advice_note = ("建议级别: 观察——预测计分卡冷启动中，"
                                   "每日自动核对 TOP 列表次日实际表现，"
                                   "实证达标前不给出买卖建议")
        except Exception:
            pass

        out = ["", f"【相对强弱预测 TOP{top_n} · 截至 {rep.get('last_date', '')}】",
               "-" * 60]
        out.append("预测口径: 60日动量 − 行业均值（行业中性），全池横截面排序")
        out.append(conf)
        if advice_note:
            out.append(advice_note)
        for i, p in enumerate(pred[:top_n], 1):
            out.append(f"  {i:>2}. {p['symbol']} {p.get('name', ''):<8} "
                       f"[{p.get('sector', '')}] 60日动量 {p.get('mom_60d_pct', 0):+.1f}%")
        out.append("-" * 60)
        return out

    def _sector_rotation_lines(self) -> List[str]:
        """【行业轮动预测 TOP】—— 周频行业动量共识预测（系统最强验证线）。

        回测背景: 扣ETF成本后超额年化+10~23%, 12/12参数全正, OOS 12/12同向
        (data/sector_rotation_report.json)。市况依赖: 趋势市强、震荡市弱。
        统计数字动态读报告/计分卡, 禁止硬编码。
        """
        try:
            root = Path(__file__).resolve().parent.parent
            rep = json.loads((root / "data" / "sector_rotation_report.json")
                             .read_text(encoding="utf-8"))
        except Exception:
            return []
        pred = rep.get("prediction") or []
        if not pred:
            return []
        total = rep.get("total_cells", 12)

        # 建议级别（动态读计分卡）
        advice = "建议级别: 未见计分卡数据"
        try:
            sc = rep.get("scorecard", {})
            level = sc.get("advice_level", "OBSERVE")
            nc = sc.get("n_checked", 0)
            roll = sc.get("rolling", {})
            if level == "BUY_LEAN":
                advice = (f"🎯 建议级别: 买入倾向——滚动 {roll.get('n')} 周预测超额 "
                          f"{roll.get('mean', 0):+.3f}%/周 t={roll.get('t', 0):+.2f}"
                          f"命中 {roll.get('hit_rate', 0):.0%}，"
                          "TOP3 行业可作行业 ETF 买入候选")
            elif level == "AVOID":
                advice = (f"⚠️ 建议级别: 回避提示——滚动预测超额 "
                          f"{roll.get('mean', 0):+.3f}%/周 t={roll.get('t', 0):+.2f} 显著为负")
            else:
                advice = (f"建议级别: 观察——已核对 {nc} 周预测，滚动超额 "
                          f"{roll.get('mean', 0):+.3f}%/周 t={roll.get('t', 0):+.2f}"
                          f"（滚动 t 过 ±2 才升级/降级）")
        except Exception:
            pass

        # 官方指数口径复核（零幸存者偏差终审）——动态读，禁止硬编码
        final_verdict = ""
        try:
            sw = json.loads((root / "data" /
                             "sector_rotation_swindex_old27_report.json")
                            .read_text(encoding="utf-8"))
            rs = sw.get("results", [])
            if rs:
                wins = sum(1 for r in rs if r.get("t", 0) > 2)
                excs = sorted(r.get("exc", 0) for r in rs)
                n = len(excs)
                med = excs[n // 2] if n % 2 else (excs[n // 2 - 1]
                                                  + excs[n // 2]) / 2
                final_verdict = (
                    f"⚠️ 终审: 官方申万指数口径({sw.get('start', '')[:7]}~"
                    f"{sw.get('end', '')[:7]}, 零幸存者偏差)复核——"
                    f"{wins}/{n} 组显著为正, 中位超额 {med:+.1f}%/年。"
                    "行业轮动在可执行口径(行业ETF)上无正超额证据; "
                    "合成池口径的历史正超额已判定为幸存者偏差产物。"
                    "本段仅作池内相对观察, 勿按此买行业ETF。")
        except Exception:
            final_verdict = ""

        out = ["", f"【行业轮动预测 TOP3 · 截至 {rep.get('last_date', '')}】",
               "-" * 60]
        out.append("预测口径: 行业动量周频, 12组参数(回看×TOP数)共识票数")
        if final_verdict:
            out.append(final_verdict)
        out.append(advice)
        for i, p in enumerate(pred[:3], 1):
            out.append(f"  {i}. {p['sector']}  "
                       f"共识 {p.get('votes', 0)}/{total} "
                       f"平均排名 {p.get('avg_rank', '-')}")
        out.append("  (共识票数分散=当前行业动量信号不凝聚, 降低参考权重)")
        out.append("-" * 60)
        return out

    def _industry_strength_lines(self, predictions, top_n: Optional[int] = None,
                                 with_anomaly: bool = True) -> List[str]:
        """行业相对强弱 + 行业内异动 —— L1 工具化的核心价值段落。

        为什么加: 个股方向预测已被证伪/未检出，但「谁比谁强」的**横向比较**
        不依赖任何预测能力，是纯描述性信息，对看盘有确定价值。这也是把系统
        从"猜方向"改造成"看得清楚"的关键一步。
        """
        if not predictions:
            return []
        groups: Dict[str, list] = {}
        for p in predictions:
            sec = (getattr(getattr(p, "stock", None), "sector", "") or "未分类")
            groups.setdefault(sec, []).append(p)
        if len(groups) < 3:
            return []      # 行业太少，横向排名无意义

        rows = []
        for sec, g in groups.items():
            chg = [x.change_percent for x in g if x.change_percent is not None]
            sc = [x.final_score for x in g if x.final_score is not None]
            if not chg:
                continue
            rows.append({
                "sec": sec, "n": len(g), "chg": sum(chg) / len(chg),
                "score": (sum(sc) / len(sc)) if sc else 0.0,
                "lead": max(g, key=lambda x: (x.change_percent or -99)),
                "lag": min(g, key=lambda x: (x.change_percent or 99)),
            })
        if not rows:
            return []
        rows.sort(key=lambda r: -r["chg"])
        if top_n is not None and len(rows) > 2 * top_n:
            rows = rows[:top_n] + [{"sep": True}] + rows[-top_n:]

        out = ["", "【行业相对强弱 · 横向比较（描述事实，非预测）】", "-" * 60]
        out.append(f"{'#':<4}{'行业':<12}{'只数':>4}{'均涨跌%':>10}{'均评分':>7}"
                   f"  {'领涨':<18}{'领跌':<18}")
        for i, r in enumerate(rows, 1):
            if r.get("sep"):
                out.append(" " * 4 + "· · · （中间行业略，完整版见文件）")
                continue
            lead = f"{r['lead'].stock.name} {r['lead'].change_percent:+.1f}%"
            lag = f"{r['lag'].stock.name} {r['lag'].change_percent:+.1f}%"
            out.append(f"{i:<5}{r['sec']:<12}{r['n']:>4}{r['chg']:>+10.2f}"
                       f"{r['score']:>7.1f}  {lead:<20}{lag:<20}")

        if not with_anomaly:
            out.append("-" * 60)
            return out

        moved = []
        for r in rows:
            if r.get("sep"):
                continue
            for x in groups[r["sec"]]:
                if x.change_percent is None:
                    continue
                d = x.change_percent - r["chg"]
                if abs(d) >= 3.0:
                    moved.append((d, x, r["sec"], r["chg"]))
        out.append("")
        out.append("【行业内异动 · 偏离所属行业均值 ≥3%】")
        if moved:
            moved.sort(key=lambda t: -abs(t[0]))
            cap = 5 if top_n is not None else 12
            for d, x, sec, avg in moved[:cap]:
                tag = "强于" if d > 0 else "弱于"
                out.append(f"- {x.stock.name}({x.stock.symbol}) {x.change_percent:+.1f}%"
                           f" vs {sec}均 {avg:+.1f}% → {tag}行业 {abs(d):.1f}%")
        else:
            out.append("- 无显著偏离")
        out.append("-" * 60)
        return out

    def _individual_strength_lines(self, predictions, top_n: Optional[int] = None) -> List[str]:
        """个股相对强弱排名（全池横向比较，纯观察、非信号）。

        L1 工具化的另一半价值: 个股方向无法预测，但「今天谁强谁弱」是可观测的
        真实事实。给出全池最强/最弱两端，帮助看盘时快速定位焦点，不做任何方向建议。
        """
        valid = [p for p in predictions
                 if p.change_percent is not None and p.stock.sector != "未分类"]
        if len(valid) < 10:
            return []
        ranked = sorted(valid, key=lambda x: -x.change_percent)
        n_top = top_n if top_n is not None else 12
        n_bot = top_n if top_n is not None else 12
        top, bot = ranked[:n_top], ranked[::-1][:n_bot]
        out = ["", "【个股相对强弱 · 全池观察（事实，非建议）】", "-" * 60]
        out.append(f"{'#':<4}{'名称':<12}{'代码':<10}{'行业':<10}{'涨跌%':>8}{'评分':>6}")
        for i, p in enumerate(top, 1):
            out.append(f"{i:<5}{p.stock.name:<12}{p.stock.symbol:<10}"
                       f"{p.stock.sector:<10}{p.change_percent:>+8.2f}{p.final_score:>6.1f}")
        if top_n is None or len(ranked) > 2 * top_n:
            out.append(" " * 4 + "· · · · · · · · · · · · · · · · · · · · · · · ·")
        for i, p in enumerate(bot, 1):
            out.append(f"{i:<5}{p.stock.name:<12}{p.stock.symbol:<10}"
                       f"{p.stock.sector:<10}{p.change_percent:>+8.2f}{p.final_score:>6.1f}")
        out.append("-" * 60)
        return out

    def _morning_iteration_briefing_lines(self, analysis_type: str) -> List[str]:
        if analysis_type != "morning":
            return []
        brief_path = self.data_dir / "iteration_briefing.txt"
        if not brief_path.exists():
            return []
        brief_txt = brief_path.read_text(encoding="utf-8").strip()
        if not brief_txt:
            return []
        return [
            "",
            "【持续迭代简报】",
            "-" * 60,
            *brief_txt.splitlines(),
            "-" * 60,
        ]
    
    def generate_prediction_report(self, predictions: List[PredictionResult], analysis_type: str) -> str:
        """生成预测报告"""
        
        report_lines = []
        is_morning = (analysis_type == "morning")
        if is_morning:
            report_lines.append("【A股早盘信号观察 · 研究参考】")
        else:
            report_lines.append(f"【A股{get_analysis_type_name(analysis_type)}预测报告】")
        report_lines.append("=" * 60)
        report_lines.append(f"预测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"分析类型: {get_analysis_type_name(analysis_type)}")
        report_lines.append("=" * 60)
        if is_morning:
            report_lines.append(_alpha_evidence_disclaimer())
        report_lines.extend(self._morning_iteration_briefing_lines(analysis_type))
        report_lines.extend(self._industry_strength_lines(predictions))
        report_lines.extend(self._individual_strength_lines(predictions))

        th = ConfigManager.get_signal_thresholds()
        b_line, h_line = th["buy"], th["hold"]
        # 分类预测结果
        buy_predictions = [p for p in predictions if p.final_score >= b_line]
        sell_predictions = [p for p in predictions if p.final_score < h_line]
        hold_predictions = [p for p in predictions if h_line <= p.final_score < b_line]
        
        # 公式信号（观察用，非买卖建议；系统定位见顶部风险提示）
        if buy_predictions:
            report_lines.append(f"\n【公式买入信号 · 观察(非建议)】 ({len(buy_predictions)}只)")
            for i, pred in enumerate(buy_predictions, 1):
                self._add_prediction_detail(i, pred, report_lines)

        # 卖出预测
        if sell_predictions:
            report_lines.append(f"\n【公式卖出信号 · 观察(非建议)】 ({len(sell_predictions)}只)")
            for i, pred in enumerate(sell_predictions, 1):
                self._add_prediction_detail(i, pred, report_lines)

        # 持有预测
        if hold_predictions:
            report_lines.append(f"\n【公式持有信号 · 观察(非建议)】 ({len(hold_predictions)}只)")
            for i, pred in enumerate(hold_predictions, 1):
                self._add_prediction_detail(i, pred, report_lines)
        
        # 预测统计
        self._add_prediction_statistics(predictions, buy_predictions, sell_predictions, hold_predictions, report_lines)
        
        return "\n".join(report_lines)
    
    def generate_summary_report(self, summary: SummaryReport,
                                 predictions: Optional[List[PredictionResult]] = None,
                                 compact: bool = False) -> str:
        """生成总结报告。

        compact=True: 企微推送版，受 2048 字节硬限约束——精简免责声明 + 截断排名
        （仅最强/最弱两端），保证送达；完整内容见落盘文件。
        compact=False: 完整版（落盘文件用），含全部行业与个股排名。
        """
        report_lines = []
        report_lines.append(f"【A股{get_analysis_type_name(summary.analysis_type)}总结报告】")
        report_lines.append("=" * 60)
        report_lines.append(f"总结时间: {summary.report_time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"分析类型: {get_analysis_type_name(summary.analysis_type)}")
        report_lines.append("=" * 60)

        # 诚实声明（顶部，先于一切信号）
        report_lines.append(_alpha_evidence_disclaimer_short() if compact
                            else _alpha_evidence_disclaimer())

        # 核心预测输出：行业中性相对强弱 TOP + 行业轮动共识（预测层）
        report_lines.extend(self._rs_prediction_lines(
            top_n=5 if compact else 10))
        report_lines.extend(self._sector_rotation_lines())
        # 交易成本账：量化"不买卖"的价值（compact 版同样显示，这是最重要的提示）
        report_lines.extend(self._trading_cost_lines())
        # 配置指导：唯一有历史验证的买卖框架（被动配置+再平衡）
        report_lines.extend(self._allocation_lines())
        # 今日操作指令：系统下达明确买卖指令，照做即可
        report_lines.extend(self._portfolio_action_lines())
        # 实盘跟踪：按系统执行的组合收益 vs 基准（系统值不值的真值，每日更新）
        report_lines.extend(self._track_lines())

        # 横向比较：行业相对强弱 + 行业内异动 + 个股相对强弱（描述层，辅助验证预测）
        if predictions:
            if compact:
                report_lines.extend(self._industry_strength_lines(predictions, top_n=5, with_anomaly=False))
                report_lines.extend(self._individual_strength_lines(predictions, top_n=5))
            else:
                report_lines.extend(self._industry_strength_lines(predictions))
                report_lines.extend(self._individual_strength_lines(predictions))

        # 持续迭代简报（紧凑推送版省略，受企微字节限约束；完整版见落盘文件）
        if not compact:
            report_lines.extend(self._morning_iteration_briefing_lines(summary.analysis_type))

        # 推荐总结（紧凑版只给计数，受企微字节限约束；完整逐只列表见落盘文件）
        if compact:
            nb, ns, nh = (len(summary.buy_recommendations),
                          len(summary.sell_recommendations),
                          len(summary.hold_recommendations))
            report_lines.append("\n【公式信号计数 · 观察(非建议)】")
            report_lines.append(f"买入信号 {nb}只 | 卖出信号 {ns}只 | 持有信号 {nh}只"
                                 f"（无 alpha，仅供研究参考，完整列表见报告文件）")
        else:
            self._add_recommendation_summary(summary, report_lines)

            # 以下为完整版内容（紧凑推送版省略，受企微 2048 字节限约束）
            # 市场概况
            self._add_market_overview_summary(summary.market_overview, report_lines)

            # 行业分析
            self._add_sector_analysis_summary(summary.sector_analysis, report_lines)

            # 风险提示
            if summary.risk_alerts:
                report_lines.append(f"\n【风险提示】")
                for i, alert in enumerate(summary.risk_alerts, 1):
                    report_lines.append(f"{i}. {alert}")

            # 下一步行动
            if summary.next_actions:
                report_lines.append(f"\n【下一步行动】")
                for i, action in enumerate(summary.next_actions, 1):
                    report_lines.append(f"{i}. {action}")

        return "\n".join(report_lines)
    
    def save_results(self, predictions: List[PredictionResult], summary: SummaryReport, 
                    prediction_report: str, summary_report: str, analysis_type: str) -> Dict[str, str]:
        """保存结果到文件"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存预测数据
        prediction_data = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': analysis_type,
            'data_provider': 'openclaw',
            'predictions': [pred.to_dict() for pred in predictions],
            'prediction_count': len(predictions),
            'calibration': {
                'signal_thresholds': ConfigManager.get_signal_thresholds(),
                'score_weights': ConfigManager.get_score_weights(),
                'accuracy_tuning': ConfigManager.get_accuracy_tuning(),
            },
        }
        
        prediction_file = self.data_dir / f"predictions_{analysis_type}_{timestamp}.json"
        with open(prediction_file, 'w', encoding='utf-8') as f:
            json.dump(prediction_data, f, ensure_ascii=False, indent=2)
        
        # 保存总结数据
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': analysis_type,
            'summary': summary.to_dict(),
            'report_text': summary_report
        }
        
        summary_file = self.data_dir / f"summary_{analysis_type}_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        # 保存预测报告文本
        prediction_report_file = self.reports_dir / f"prediction_report_{analysis_type}_{timestamp}.txt"
        with open(prediction_report_file, 'w', encoding='utf-8') as f:
            f.write(prediction_report)
        
        # 保存总结报告文本
        summary_report_file = self.reports_dir / f"summary_report_{analysis_type}_{timestamp}.txt"
        with open(summary_report_file, 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        return {
            'prediction_data': str(prediction_file),
            'summary_data': str(summary_file),
            'prediction_report': str(prediction_report_file),
            'summary_report': str(summary_report_file)
        }
    
    def _add_prediction_detail(self, index: int, pred: PredictionResult, report_lines: List[str]):
        """添加预测详情"""
        report_lines.append(f"{index}. {pred.stock.name} ({pred.stock.symbol})")
        report_lines.append(f"   行业: {pred.stock.sector} | 综合评分: {pred.final_score}/10")
        report_lines.append(f"   当前价: ¥{pred.current_price:.2f} ({pred.change_percent:+.2f}%)")
        report_lines.append(f"   信号: {pred.signal} | 信心度: {pred.confidence}%")
        report_lines.append(f"   技术面: {pred.technical_score}/10 | 基本面: {pred.fundamental_score}/10")
        report_lines.append(f"   情绪面: {pred.sentiment_score}/10 | 行业面: {pred.sector_score}/10")
        report_lines.append(f"   预测理由: {'; '.join(pred.reasons)}")
        report_lines.append("")
    
    def _add_prediction_statistics(self, predictions: List[PredictionResult], 
                                 buy: List[PredictionResult], sell: List[PredictionResult], 
                                 hold: List[PredictionResult], report_lines: List[str]):
        """添加预测统计"""
        
        th = ConfigManager.get_signal_thresholds()
        report_lines.append(f"\n【预测统计】（档位: 买入≥{th['buy']} 持有≥{th['hold']} 卖出<{th['hold']}）")
        report_lines.append(f"总预测股票数: {len(predictions)}")
        report_lines.append(f"买入预测: {len(buy)}只")
        report_lines.append(f"卖出预测: {len(sell)}只") 
        report_lines.append(f"持有预测: {len(hold)}只")
        
        if buy:
            avg_buy_score = sum(p.final_score for p in buy) / len(buy)
            report_lines.append(f"买入预测平均评分: {avg_buy_score:.1f}分")
        
        if sell:
            avg_sell_score = sum(p.final_score for p in sell) / len(sell)
            report_lines.append(f"卖出预测平均评分: {avg_sell_score:.1f}分")
        
        if hold:
            avg_hold_score = sum(p.final_score for p in hold) / len(hold)
            report_lines.append(f"持有预测平均评分: {avg_hold_score:.1f}分")
        
        # 行业分布
        buy_sectors = list(set(p.stock.sector for p in buy))
        sell_sectors = list(set(p.stock.sector for p in sell))
        
        if buy_sectors:
            report_lines.append(f"买入预测行业: {', '.join(buy_sectors)}")
        if sell_sectors:
            report_lines.append(f"卖出预测行业: {', '.join(sell_sectors)}")
    
    def _add_recommendation_summary(self, summary: SummaryReport, report_lines: List[str]):
        """添加推荐总结"""
        
        report_lines.append(f"\n【信号分档观察 · 非买卖建议】")
        
        if summary.buy_recommendations:
            report_lines.append(f"偏多信号 ({len(summary.buy_recommendations)}只):")
            for i, rec in enumerate(summary.buy_recommendations, 1):
                report_lines.append(f"  {i}. {rec.stock.name} ({rec.stock.symbol}) - 评分:{rec.final_score}")
        
        if summary.sell_recommendations:
            report_lines.append(f"偏空信号 ({len(summary.sell_recommendations)}只):")
            for i, rec in enumerate(summary.sell_recommendations, 1):
                report_lines.append(f"  {i}. {rec.stock.name} ({rec.stock.symbol}) - 评分:{rec.final_score}")
        
        if summary.hold_recommendations:
            report_lines.append(f"中性信号 ({len(summary.hold_recommendations)}只):")
            for i, rec in enumerate(summary.hold_recommendations, 1):
                report_lines.append(f"  {i}. {rec.stock.name} ({rec.stock.symbol}) - 评分:{rec.final_score}")
    
    def _add_market_overview_summary(self, market_overview: Dict, report_lines: List[str]):
        """添加市场概况总结"""
        
        report_lines.append(f"\n【市场概况】")
        report_lines.append(f"总股票数: {market_overview['total_stocks']}")
        report_lines.append(f"偏多信号: {market_overview['buy_count']}只 | 平均评分: {market_overview['avg_buy_score']}")
        report_lines.append(f"偏空信号: {market_overview['sell_count']}只 | 平均评分: {market_overview['avg_sell_score']}")
        report_lines.append(f"中性信号: {market_overview['hold_count']}只 | 平均评分: {market_overview['avg_hold_score']}")
        report_lines.append(f"市场情绪: {market_overview['market_sentiment']}")
        if market_overview.get("market_status"):
            vol = market_overview.get("market_volatility")
            trend = market_overview.get("market_trend_strength")
            if vol is None or trend is None:
                detail = "状态未知（行情源瞬时抖动，非震荡市）"
            else:
                detail = f"vol≈{vol*100:.0f}%, 20日动量≈{trend*100:+.1f}%"
            report_lines.append(
                f"市场状态: {market_overview['market_status_desc']} ({detail})"
            )
    
    def _add_sector_analysis_summary(self, sector_analysis: Dict, report_lines: List[str]):
        """添加行业分析总结"""
        
        if sector_analysis:
            report_lines.append(f"\n【行业分析】")
            for sector, data in sector_analysis.items():
                report_lines.append(f"{sector}: 平均评分{data['avg_score']} ({data['trend']}, {data['stock_count']}只股票)")


# ==================== 主分析器 ====================

class StockAnalyzer:
    """股票分析器 - 遵循"先预测再总结"理念"""
    
    def __init__(
        self,
        base_dir: Optional[str] = None,
        data_provider: Optional[StockDataProvider] = None,
    ):
        root = base_dir or _default_stock_system_root()
        ConfigManager.reload_stock_pool()
        ConfigManager.reload_calibration()
        self.prediction_engine = PredictionEngine(data_provider=data_provider)
        self.summary_engine = SummaryEngine()
        self.report_generator = ReportGenerator(root)
        self.base_dir = Path(root)
    
    def analyze(self, analysis_type: str = "evening") -> Dict[str, any]:
        """执行完整分析流程 - 先预测再总结"""
        ConfigManager.reload_calibration()
        print(f"🚀 开始{get_analysis_type_name(analysis_type)}分析...")

        # ── 前置：检查 OpenClaw Gateway 健康状态 ──
        self._print_gateway_status()

        print("📊 第一步: 生成个股预测...")
        
        # 第一步: 预测阶段 - 对每只股票进行独立预测（股票池见 config/stock_pool.json）
        universe = ConfigManager.get_analysis_stock_slice(analysis_type)
        predictions = self.prediction_engine.predict_portfolio(universe, analysis_type)
        
        print(f"✅ 预测完成，共{len(predictions)}只股票")
        print("📋 第二步: 生成总结报告...")
        
        # 第二步: 总结阶段 - 基于所有预测结果生成总结
        # 把路线 A 闸门状态并入 market_status，供报告如实展示
        ms = dict(self.prediction_engine._market_status or {})
        ms["index_trend_gate"] = getattr(self.prediction_engine, "_market_gate", "normal")
        summary = self.summary_engine.generate_summary(
            predictions, analysis_type, market_status=ms
        )
        
        print("✅ 总结报告生成完成")
        print("📝 第三步: 生成文本报告...")
        
        # 第三步: 生成文本报告
        prediction_report = self.report_generator.generate_prediction_report(predictions, analysis_type)
        summary_report = self.report_generator.generate_summary_report(summary, predictions, compact=False)
        # 企微推送版（受 2048 字节硬限，须精简；完整版见落盘文件）
        summary_report_compact = self.report_generator.generate_summary_report(summary, predictions, compact=True)

        # ── #3 行情源瞬时失败未纳入的个股：报告正文标注（覆盖 .txt 与企微推送）──
        skipped = list(getattr(self.prediction_engine, "_skipped_stocks", []) or [])
        if skipped:
            skip_note = (
                "\n\n【行情源瞬时失败未纳入】\n  "
                + "；".join(skipped)
                + f"\n  （共 {len(skipped)} 只：因行情源偶发抖动缺数据未分析，非系统剔除，"
                  f"次日行情恢复即自动回归；不影响其余信号）"
            )
            summary_report = summary_report.rstrip() + skip_note
            summary_report_compact = summary_report_compact.rstrip() + skip_note
        
        print("✅ 文本报告生成完成")
        print("💾 第四步: 保存结果...")
        
        # 第四步: 保存所有结果
        saved_files = self.report_generator.save_results(
            predictions, summary, prediction_report, summary_report, analysis_type
        )
        
        print("✅ 结果保存完成")
        
        return {
            'predictions': predictions,
            'summary': summary,
            'prediction_report': prediction_report,
            'summary_report': summary_report,
            'summary_report_compact': summary_report_compact,
            'saved_files': saved_files,
            'analysis_type': analysis_type,
            'total_stocks': len(universe),
        }
    
    @staticmethod
    def _print_gateway_status() -> None:
        """启动前检测 OpenClaw Gateway 是否可达，提前给出提示。"""
        import os as _os
        if _os.environ.get("STOCK_SKIP_AGENT", "").lower() in ("1", "true", "yes"):
            print("⚡ STOCK_SKIP_AGENT=1，跳过 Agent，直接使用 akshare 拉取行情。")
            return
        import urllib.request
        try:
            req = urllib.request.Request("http://127.0.0.1:18789/health")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    print("🟢 OpenClaw Gateway 正常，将使用 Agent 拉取行情。")
                    return
        except Exception:
            pass
        print("🔴 OpenClaw Gateway 不可达！将自动降级到 akshare 直取行情。")
        print("   如需恢复 Agent 拉取，请确认 openclaw gateway 已启动。")


# ==================== 主函数 ====================

def main():
    """主函数"""
    
    print("🚀 启动A股分析系统 (符合\"先预测再总结\"理念)")
    print("=" * 70)
    print(f"系统时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 创建分析器
    analyzer = StockAnalyzer()
    
    # 执行分析（默认收盘分析）
    result = analyzer.analyze("evening")
    
    # 显示结果
    print("\n📊 分析结果:")
    print("=" * 70)
    print(result['summary_report'])
    
    print("\n💾 文件保存位置:")
    for file_type, file_path in result['saved_files'].items():
        print(f"  {file_type}: {file_path}")
    
    print("\n✅ 分析完成！")
    print("🎯 系统特点:")
    print("  ✓ 先预测: 对每只股票进行独立预测分析")
    print("  ✓ 再总结: 基于所有预测结果生成综合总结")
    print("  ✓ 结构化: 清晰的预测->总结流程")
    print("  ✓ 文件管理: 所有文件统一保存在指定目录")


if __name__ == "__main__":
    main()