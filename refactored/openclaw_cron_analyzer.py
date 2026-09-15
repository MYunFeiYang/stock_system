#!/usr/bin/env python3
"""
OpenClaw定时任务专用的股票分析系统
支持不同分析类型：morning, afternoon, evening, weekly
以及复盘类：reconcile、day_review；**post_close** 为二者连续执行（先复盘再汇总与自校准，供定时任务一次跑完）。
"""
from __future__ import annotations

import pathlib as _pl
_orig_path_mkdir = _pl.Path.mkdir
def _broker_safe_mkdir(self, mode=0o777, parents=False, exist_ok=False):
    try:
        return _orig_path_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)
    except PermissionError:
        if exist_ok and self.exists():
            return None  # WorkBuddy broker 不支持 exist_ok 语义，已存在目录忽略
        raise
_pl.Path.mkdir = _broker_safe_mkdir

import os
import sys
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path

# 导入预测和总结引擎
sys.path.append(os.path.dirname(__file__))
from predict_then_summarize import StockAnalyzer
from data_providers import get_analysis_type_name


class _LogTee:
    """把 stdout/stderr 同时写回原终端（供 gateway 捕获）和本地日志文件。

    解决可观测性缺口：此前 cron 运行诊断只存在 gateway 内存（lastDiagnosticSummary，
    截断且被下次运行覆盖），logs/ 目录长期无 cron 运行日志，排障只能靠 cron get。
    """

    def __init__(self, stream, logf):
        self._stream = stream
        self._logf = logf

    def write(self, data):
        try:
            self._stream.write(data)
        except Exception:
            pass
        try:
            self._logf.write(data)
            self._logf.flush()
        except Exception:
            pass
        return len(data)

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        try:
            self._logf.flush()
        except Exception:
            pass


def _install_log_tee(base_dir: Path, now: datetime) -> Path | None:
    """把本次 cron 运行的全部 stdout/stderr 落盘 logs/cron_YYYY-MM-DD.log（append）。

    morning/evening/post_close 等共用同日文件；返回日志路径，失败返回 None（不阻断）。
    """
    try:
        log_dir = base_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"cron_{now.strftime('%Y-%m-%d')}.log"
        logf = open(log_path, "a", encoding="utf-8")
        sys.stdout = _LogTee(sys.stdout, logf)
        sys.stderr = _LogTee(sys.stderr, logf)
        return log_path
    except Exception as e:
        print(f"⚠️ 日志落盘初始化失败（不影响主流程）: {e}")
        return None


def _is_trading_day(dt: datetime) -> bool:
    """A 股交易日判断。

    周末直接返回 False；工作日用 akshare 交易日历校验（含节假日）。
    交易日历拉取失败时回退为 True（假定交易日），避免网络抖动阻塞运行。
    """
    if dt.weekday() >= 5:  # 5=周六, 6=周日
        return False
    try:
        import akshare as ak
        cal = ak.tool_trade_date_hist_sina()
        if cal is not None and "trade_date" in cal.columns:
            trade_dates = {str(d)[:10] for d in cal["trade_date"].tolist()}
            return dt.strftime("%Y-%m-%d") in trade_dates
    except Exception:
        pass
    return True


def _read_strategy_degraded_text(base_dir: str) -> str:
    """读取策略失效预警文本（若策略处于 degraded）。返回告警字符串，否则空串。"""
    status_path = Path(base_dir) / "data" / "strategy_status.json"
    if not status_path.exists():
        return ""
    try:
        st = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if st.get("status") != "degraded":
        return ""
    rates = st.get("recent_directional_hit_rates", [])
    return (
        f"\n⚠️⚠️ 策略失效预警 ⚠️⚠️\n"
        f"   连续 {st.get('window')} 个交易日方向性一致率 {rates} < {st.get('threshold')}\n"
        f"   近期策略疑似失效，早报信号参考价值低，建议观望/减仓。"
    )


def _maybe_warn_strategy_degraded(base_dir: str) -> str:
    """若策略处于 degraded，早盘打印醒目警告（不篡改保存信号，供复盘继续统计校准）。
    返回告警文本（供推送复用），无则空串。"""
    alert = _read_strategy_degraded_text(base_dir)
    if alert:
        print(alert)
    return alert


def _latest_report(base_dir: str, prefix: str, day: str):
    """找 reports/ 目录下当日最新 {prefix}_{day}_*.txt 报告路径。"""
    d = Path(base_dir) / "reports"
    if not d.exists():
        return None
    fs = sorted(d.glob(f"{prefix}_{day}_*.txt"))
    return fs[-1] if fs else None


def _push_via_webhook(text: str, title: str = "") -> bool:
    """企微群机器人 webhook 直发（fire-and-forget，绕过 aibot 5s ack 超时）。

    需环境变量 WECOM_WEBHOOK_URL（企微群机器人 Webhook 地址）。
    直接 HTTP POST 到企微服务器，不经过 gateway / aibot，无 5s ack 约束，
    是定时冷启动场景下的确定性通道。返回是否成功。

    字节策略: 企微 text 消息硬限 2048 字节。若内容超限，自动改用 markdown
    消息（限 4096 字节）并以代码块包裹，保留等宽对齐的表格可读性，
    确保长报告也能送达（不再因截断丢失内容）。
    """
    import urllib.request
    import urllib.error
    url = os.environ.get("WECOM_WEBHOOK_URL")
    if not url:
        return False
    content = (title + "\n\n" + text) if title else text
    raw = content.encode("utf-8")
    if len(raw) <= 1900:
        payload = json.dumps(
            {"msgtype": "text", "text": {"content": content}},
            ensure_ascii=False,
        ).encode("utf-8")
    else:
        # markdown + 代码块：保留等宽对齐，且 4096 字节上限足够覆盖紧凑报告
        md = f"# {title}\n\n```\n{text}\n```" if title else f"```\n{text}\n```"
        payload = json.dumps(
            {"msgtype": "markdown", "markdown": {"content": md}},
            ensure_ascii=False,
        ).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", "ignore")
        try:
            j = json.loads(body)
            if int(j.get("errcode", -1)) == 0:
                print("✅ 企微 webhook 推送成功")
                return True
            print(f"⚠️ 企微 webhook 返回非0: {body[:200]}")
        except Exception:
            print(f"✅ 企微 webhook 推送成功(响应 {body[:80]})")
            return True
    except Exception as e:
        print(f"⚠️ 企微 webhook 推送异常: {e}")
    return False


def push_to_wecom(text: str, title: str = "") -> bool | None:
    """推送文本到企微(thinkway)。

    三态返回：
      True  = webhook 已配且发送成功
      False = webhook 已配但发送失败（需排查 URL/网络）
      None  = webhook 未配置（推送由外部通道负责，如 WorkBuddy 定时任务企微 bot 同步）

    背景：openclaw 2026.8.1 的 `message send --channel wecom` CLI 不再路由插件 channel
    （wecom 由插件注册，不在 CLI 核心 channel 枚举里），原主通道已失效；且 command 型
    cron 的 delivery(announce) 不投递脚本 stdout（仅投递 agent 会话最终文本），故
    cron 原生投递对脚本型任务无效。

    当前可用主通道：群机器人 webhook 直发（WECOM_WEBHOOK_URL）。该路径是确定性 HTTP
    POST，完全独立于 openclaw 的插件 channel 路由，不受上述 CLI 限制影响。脚本已内置
    _push_via_webhook 实现；只需在环境/ cron env 中配置 WECOM_WEBHOOK_URL 即可启用。
    未配置时返回 None（表示"未尝试、由外部通道推送"，不记为失败）。
    """
    import sys
    url = os.environ.get("WECOM_WEBHOOK_URL")
    if url:
        return _push_via_webhook(text, title)
    # webhook 未配置：推送由外部通道负责（如 WorkBuddy 定时任务企微 bot 同步 /
    # openclaw gateway announce），本函数不视为失败，返回 None 让调用方跳过记录。
    return None


def _record_push_status(base_dir: str, ok: bool | None) -> None:
    """记录推送成败到 data/push_status.json（跨运行累计连续失败，供早盘告警）。

    ok=None 表示 webhook 未配置、推送由外部通道负责，本函数跳过不记录。
    """
    if ok is None:
        return  # 外部通道（如 WorkBuddy 定时任务企微 bot 同步）负责推送，脚本未尝试
    p = Path(base_dir) / "data" / "push_status.json"
    try:
        st = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        st = {}
    iso = datetime.now().isoformat()
    if ok:
        st["consecutive_failures"] = 0
        st["last_success_at"] = iso
    else:
        st["consecutive_failures"] = int(st.get("consecutive_failures", 0)) + 1
        st["last_fail_at"] = iso
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _warn_if_push_repeatedly_failing(base_dir: str) -> None:
    """若企微推送（webhook 直发模式）已连续多次失败，打印醒目告警（不阻断本次运行）。

    注意：当且仅当 WECOM_WEBHOOK_URL 已配置时推送才可能成功；未配则不在此告警
    （避免误报），管道健康改由 _emit_pipeline_health 经 webhook 配置自检覆盖。
    """
    if not os.environ.get("WECOM_WEBHOOK_URL"):
        return
    p = Path(base_dir) / "data" / "push_status.json"
    if not p.exists():
        return
    try:
        st = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return
    cf = int(st.get("consecutive_failures", 0))
    if cf >= 3:
        print(f"\n🚨 企微推送已连续 {cf} 次失败！请检查 WECOM_WEBHOOK_URL 群机器人"
              f" webhook 是否有效（是否被撤销/过期）。本次仍会尝试推送。")


def _emit_pipeline_health() -> None:
    """早报开头自检推送管道：说明当前推送方式。

    脚本不再负责推送（webhook 未配时由外部通道如 WorkBuddy 定时任务企微 bot 同步
    自动路由到企微；webhook 配了则双保险直发）。本函数仅打印信息，不阻断主流程。
    """
    webhook = os.environ.get("WECOM_WEBHOOK_URL")
    if webhook:
        print("✅ 推送管道：WECOM_WEBHOOK_URL 已配置，早报将经群机器人直发企微（双保险）。")
    else:
        print(
            "ℹ️  推送管道：脚本不直发企微（WECOM_WEBHOOK_URL 未配置），"
            "报告将由外部通道（如 WorkBuddy 定时任务企微 bot 同步）自动路由推送。"
        )


def main():
    """主函数 - 支持命令行参数指定分析类型"""

    # 行缓冲：管道/被 agentTurn 轮询时 stdout 实时可见，
    # 避免"块缓冲长时间零输出"被调度器误判挂起而 SIGTERM（2026-09-14 事故）
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    # 获取分析类型参数
    analysis_type = sys.argv[1] if len(sys.argv) > 1 else "evening"
    
    # 验证分析类型
    valid_types = [
        'morning',
        'afternoon',
        'evening',
        'weekly',
        'reconcile',
        'day_review',
        'post_close',
    ]
    if analysis_type not in valid_types:
        print(f"❌ 无效的分析类型: {analysis_type}")
        print(f"✅ 有效的类型: {', '.join(valid_types)}")
        return 1

    # 核心预测层：刷新行业中性相对强弱预测 + 行业轮动预测（fail-open，超时/失败不阻塞主流程）
    if analysis_type in ('morning', 'afternoon', 'evening', 'post_close'):
        for script, label in (("predict_relative_strength.py", "相对强弱预测"),
                              ("sector_rotation.py", "行业轮动预测")):
            try:
                t0 = time.time()
                r = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve().parent / script)],
                    capture_output=True, text=True, timeout=420,
                )
                ok = r.returncode == 0
                print(f"🔮 {label}刷新{'完成' if ok else '失败'}"
                      f" ({time.time()-t0:.0f}s)"
                      + ("" if ok else f": {r.stderr.strip().splitlines()[-1][:120] if r.stderr.strip() else 'rc='+str(r.returncode)}"))
            except subprocess.TimeoutExpired:
                print(f"🔮 {label}刷新超时(420s)，本次沿用上次结果。")
            except Exception as e:
                print(f"🔮 {label}刷新异常(忽略): {e}")


    # 非交易日跳过：避免周末/节假日无效运行 + 噪音（reconcile 无早盘文件也会告警）
    now = datetime.now()
    if not _is_trading_day(now):
        print(f"📅 今日 {now.strftime('%Y-%m-%d')} 非交易日，跳过「{analysis_type}」分析。")
        return 0

    # 基础目录：环境变量优先，否则为本仓库内 stock_system 根目录
    base_dir = Path(
        os.environ.get(
            "STOCK_SYSTEM_ROOT",
            str(Path(__file__).resolve().parent.parent),
        )
    )

    # 落盘运行日志到 logs/cron_YYYY-MM-DD.log（可观测性；gateway 内存诊断不可靠）
    _install_log_tee(base_dir, now)

    if analysis_type == "reconcile":
        from daily_cycle_review import run_reconcile
        print(f"🚀 启动A股{get_analysis_type_name(analysis_type)}")
        print("=" * 70)
        print(f"系统时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        return run_reconcile(str(base_dir))

    if analysis_type == "post_close":
        from daily_cycle_review import run_reconcile, run_day_review

        print(f"🚀 启动A股{get_analysis_type_name(analysis_type)}")
        print("=" * 70)
        print(f"系统时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print("\n── 1/2 收盘复盘（reconcile）──\n")
        rc_r = run_reconcile(str(base_dir))
        print("\n── 2/2 全日汇总与规则自校准（day_review）──\n")
        rc_d = run_day_review(str(base_dir))
        if rc_r != 0:
            print(f"\n⚠️ 复盘阶段退出码 {rc_r}（可能无早盘文件或拉价失败），仍已执行汇总。")
        if rc_d != 0:
            print(f"\n❌ 汇总阶段退出码 {rc_d}")
        # 推送收盘复盘+全日汇总到企微（脚本内自管等待+重试）
        day = now.strftime("%Y%m%d")
        report_path = _latest_report(str(base_dir), "day_review_report", day)
        review_text = ""
        if report_path:
            try:
                review_text = report_path.read_text(encoding="utf-8")
            except Exception:
                review_text = ""
        if not review_text:
            review_text = "(当日无汇总报告产出)"
        degraded_alert_pc = _read_strategy_degraded_text(str(base_dir))
        if degraded_alert_pc:
            review_text = degraded_alert_pc + "\n\n" + review_text
        _warn_if_push_repeatedly_failing(str(base_dir))
        ok = push_to_wecom(review_text, title=f"📊 A股收盘复盘与汇总 {day}")
        _record_push_status(str(base_dir), ok)
        # post_close 的核心产出是 day_review（汇总 + 自校准）。
        # reconcile 需要早盘文件，缺失属于警告场景，不应让定时任务失败。
        return rc_d

    if analysis_type == "day_review":
        from daily_cycle_review import run_day_review
        print(f"🚀 启动A股{get_analysis_type_name(analysis_type)}")
        print("=" * 70)
        print(f"系统时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        return run_day_review(str(base_dir))
    
    # 推送管道信息提示（说明当前推送方式：webhook 直发 / 外部通道自动路由）
    _emit_pipeline_health()

    print(f"🚀 启动A股{get_analysis_type_name(analysis_type)}分析")
    print("=" * 70)
    print(f"系统时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"分析类型: {get_analysis_type_name(analysis_type)}")
    print("=" * 70)
    
    # 创建分析器
    analyzer = StockAnalyzer(str(base_dir))
    
    # 执行分析
    try:
        result = analyzer.analyze(analysis_type)

        # 策略失效预警：早盘给醒目警告（不篡改保存信号，供复盘继续统计）
        degraded_alert = _maybe_warn_strategy_degraded(str(base_dir))

        print("\n📊 分析结果:")
        print("=" * 70)
        print(result['summary_report'])
        
        print("\n💾 文件保存位置:")
        for file_type, file_path in result['saved_files'].items():
            print(f"  {file_type}: {file_path}")
        
        print(f"\n✅ {get_analysis_type_name(analysis_type)}分析完成！")
        
        # 返回简洁的结果摘要
        summary = result['summary']
        buy_count = len(summary.buy_recommendations)
        sell_count = len(summary.sell_recommendations)
        hold_count = len(summary.hold_recommendations)
        analyzed = len(result['predictions'])
        
        print(f"\n📈 结果摘要:")
        print(f"  买入推荐: {buy_count}只")
        print(f"  卖出推荐: {sell_count}只") 
        print(f"  持有推荐: {hold_count}只")
        print(f"  分析股票: {analyzed}只")
        
        # 全部个股行情拉取/分析均失败
        if analyzed == 0:
            # ── pre-open 未就绪守卫（股神 8-28 拍板，技术专家落地）──
            # 早盘本应盘前产出，但 gateway 心跳延迟常把它推到盘中；若被推到 09:30 开盘前
            # （集合竞价时段行情源无最新成交价），整池被 fetch 抛错跳过属「行情未就绪」，
            # 非源故障。此场景不报 error、不推空报告，改推「跳过」提示并 exit 0，
            # 杜绝每日 pre-open 误报 error + 空推送噪音；开盘后重跑即拿真实价。
            if analysis_type == "morning" and (now.hour * 60 + now.minute) < (9 * 60 + 30):
                skipped = list(getattr(analyzer.prediction_engine, "_skipped_stocks", []) or [])
                skip_text = (
                    f"⏸️ 早盘跳过：当前 {now.strftime('%Y-%m-%d %H:%M')} 尚未开盘"
                    f"（集合竞价时段行情源无最新成交价），行情未就绪，本次不产出早报。\n"
                    f"全池 {len(skipped)} 只均因无最新价跳过（非源故障）。\n"
                    f"开盘后（09:30 起）重跑即拿真实价，无需人工干预。"
                )
                print(skip_text)
                _warn_if_push_repeatedly_failing(str(base_dir))
                ok = push_to_wecom(
                    skip_text,
                    title=f"⏸️ A股早盘跳过(行情未就绪) {now.strftime('%Y-%m-%d')}",
                )
                _record_push_status(str(base_dir), ok)
                return 0
            # 其余场景（盘中/盘后全失败，或 09:30 后源真故障）→ 视为完全失败（exit 1）
            print("❌ 无可用预测（全部股票数据缺失），本次分析失败")
            fail_text = (
                f"❌ 早盘完全失败：全部股票数据缺失，今日无任何早报信号。\n"
                f"请勿依据旧信号交易。建议检查行情源(akshare/新浪)或手动重跑。"
            )
            if degraded_alert:
                fail_text = degraded_alert + "\n\n" + fail_text
            _warn_if_push_repeatedly_failing(str(base_dir))
            ok = push_to_wecom(fail_text, title=f"❌ A股早盘完全失败 {now.strftime('%Y-%m-%d')}")
            _record_push_status(str(base_dir), ok)
            return 1

        # 部分股票缺失处理：
        #  - 显著缺失(>=5只 或 >=20%) → 视为部分失败(exit 2)，推部分失败告警
        #  - 少量缺失(<5只 且 <20%) → 属正常波动，仍推送完整早报(exit 0)，避免单只瞬缺饿死整份早报
        total = result.get('total_stocks', analyzed)
        if analyzed < total:
            missing = total - analyzed
            miss_ratio = missing / total if total else 1.0
            print(f"⚠️ 部分股票数据缺失（{missing}/{total} 只未分析，缺失率 {miss_ratio:.0%}）")
            if miss_ratio >= 0.2 or missing >= 5:
                miss_text = (
                    f"⚠️ 早盘部分失败：{missing}/{total} 只股票数据缺失，缺失率 {miss_ratio:.0%}，\n"
                    f"早报信号不完整、参考价值低，请勿据此交易。\n"
                    f"建议手动重跑（openclaw cron run 早盘任务）或等次日早盘。\n"
                    f"请检查行情源(akshare/新浪)。"
                )
                if degraded_alert:
                    miss_text = degraded_alert + "\n\n" + miss_text
                _warn_if_push_repeatedly_failing(str(base_dir))
                ok = push_to_wecom(miss_text, title=f"⚠️ A股早盘部分失败 {now.strftime('%Y-%m-%d')}")
                _record_push_status(str(base_dir), ok)
                return 2
            else:
                print(f"   缺失 {missing} 只(<20% 且 <5只)，属正常波动，推送完整早报、不另发部分失败告警。")
            # 正常波动级缺失：不 return，继续向下走正常早报推送（exit 0），避免单只瞬缺饿死整份早报

        # 推送前检查：若企微已连续多次失败，打印告警（仍尝试本次推送）
        _warn_if_push_repeatedly_failing(str(base_dir))
        # 推送早报到企微（脚本内自管等待+重试，绕过框架 5s ack 超时）
        # 用紧凑版(summary_report_compact)：受企微 2048 字节硬限，完整版见落盘文件
        push_text = result.get('summary_report_compact') or result['summary_report']
        if degraded_alert:
            push_text = degraded_alert + "\n\n" + push_text
        ok = push_to_wecom(push_text, title=f"📊 A股早盘信号观察(研究参考) {now.strftime('%Y-%m-%d')}")
        _record_push_status(str(base_dir), ok)

        return 0
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)