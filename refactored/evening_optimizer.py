#!/usr/bin/env python3
"""
收盘预测优化器 - 专门解决收盘预测准确率偏低问题
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime


class EveningPredictionOptimizer:
    """收盘预测优化器（权重/阈值与早盘共用 calibration_overrides，另保留时间衰减）。"""

    def optimize_evening_prediction(
        self,
        technical_score: float,
        fundamental_score: float,
        sentiment_score: float,
        sector_score: float,
        prediction_time: datetime,
        market_close_time: datetime = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, str, List[str]]:
        from predict_then_summarize import ConfigManager, apply_signal_margin

        if market_close_time is None:
            # 默认A股收盘时间 15:00
            market_close_time = datetime.now().replace(hour=15, minute=0, second=0, microsecond=0)
        
        # 计算时间衰减
        time_decay = self._calculate_time_decay(prediction_time, market_close_time)
        
        # 应用时间衰减到技术面和情绪面（这些受时间影响大）
        adjusted_technical = technical_score * time_decay
        adjusted_sentiment = sentiment_score * time_decay
        
        if weights is None:
            weights = ConfigManager.get_score_weights()
        final_score = (
            adjusted_technical * weights['technical'] +
            fundamental_score * weights['fundamental'] +
            adjusted_sentiment * weights['sentiment'] +
            sector_score * weights['sector']
        )
        final_score = round(final_score, 1)

        th = ConfigManager.get_signal_thresholds()
        if final_score >= th['strong_buy']:
            signal = "强烈买入"
        elif final_score >= th['buy']:
            signal = "买入"
        elif final_score >= th['hold']:
            signal = "持有"
        elif final_score >= th['sell']:
            signal = "卖出"
        else:
            signal = "强烈卖出"
        margin = float(ConfigManager.get_accuracy_tuning().get('signal_margin') or 0)
        signal = apply_signal_margin(signal, final_score, th, margin)
        
        # 生成收盘专用理由
        reasons = self._generate_evening_reasons(
            final_score, adjusted_technical, fundamental_score, 
            adjusted_sentiment, sector_score, time_decay
        )
        
        return final_score, signal, reasons
    
    def _calculate_time_decay(self, prediction_time: datetime, market_close_time: datetime) -> float:
        """计算时间衰减因子"""
        
        # 如果已经过了收盘时间，使用固定衰减
        if prediction_time >= market_close_time:
            return 0.85
        
        # 计算距离收盘的时间（小时）
        time_diff = market_close_time - prediction_time
        hours_to_close = time_diff.total_seconds() / 3600
        
        # 时间衰减公式：距离收盘越近，衰减越小
        if hours_to_close <= 0.5:  # 30分钟内
            return 0.98
        elif hours_to_close <= 1:   # 1小时内
            return 0.95
        elif hours_to_close <= 2:   # 2小时内
            return 0.90
        else:                       # 2小时以上
            return 0.85
    
    def _generate_evening_reasons(self, final_score: float, 
                                  technical_score: float, fundamental_score: float,
                                  sentiment_score: float, sector_score: float,
                                  time_decay: float) -> List[str]:
        """生成收盘专用理由"""
        
        reasons = []
        
        # 基于综合评分的基本理由
        if final_score >= 7:
            reasons.append("基本面支撑强劲，具备长期投资价值")
        elif final_score <= 3:
            reasons.append("基本面存在压力，建议谨慎观望")
        else:
            reasons.append("估值合理，适合中长期持有")
        
        # 行业分析理由
        if sector_score >= 6:
            reasons.append(f"所属行业景气度较高，政策支持明显")
        elif sector_score <= 4:
            reasons.append(f"行业面临调整压力，需关注政策变化")
        
        # 技术面分析（考虑时间衰减）
        if time_decay < 0.9:
            reasons.append("临近收盘，技术面信号需谨慎解读")
        
        if technical_score >= 7:
            reasons.append("技术面呈现积极信号，短期趋势向好")
        elif technical_score <= 3:
            reasons.append("技术面偏弱，短期存在调整压力")
        
        # 基本面强调
        if fundamental_score >= 7:
            reasons.append("公司基本面扎实，盈利能力稳定")
        elif fundamental_score <= 3:
            reasons.append("基本面存在隐忧，需关注业绩变化")
        
        # 收盘特殊理由
        if final_score >= 6:
            reasons.append("收盘前资金流入积极，市场情绪稳定")
        elif final_score <= 4:
            reasons.append("尾盘资金流出明显，短期承压")
        else:
            reasons.append("收盘阶段交易平稳，观望情绪较浓")
        
        return reasons[:3]  # 限制为3个理由