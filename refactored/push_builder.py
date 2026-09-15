#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每日推送文本组装器。

把「早盘预测 / 收盘复盘」的最终推送正文在脚本里拼装好并落盘，
自动化任务只需「跑本脚本 → 读文件 → 原样输出」，不再由模型拼模板。
这样从根上避免执行过程叙述（"先读取…""报告生成中…"）漏进用户企微。

用法：
  python push_builder.py morning   # 生成并打印早盘推送正文
  python push_builder.py close     # 生成并打印收盘推送正文

产物：
  reports/push_morning_YYYYMMDD.txt
  reports/push_close_YYYYMMDD.txt
"""
import json
import re
import sys
import glob
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prediction_accuracy_hud import build_hud_lines, compute_accuracy  # noqa: E402

TOPN = 8
DISCLAIMER = "本推送为系统自动生成的公式信号/研究记录，不构成个人投资建议。"


# ---------- 通用工具 ----------

def _today():
    return datetime.now().strftime("%Y%m%d")


def _cn_date(datestr):
    d = datetime.strptime(datestr, "%Y%m%d")
    return f"{d.month}月{d.day}日"


def _fmt_names(items):
    """items: [(name, code), ...] -> '名(码) 名(码)'"""
    return " ".join(f"{n}({c})" for n, c in items) if items else "无"


def latest_morning_report(datestr=None):
    fs = sorted(glob.glob(str(REPORTS / "summary_report_morning_*.txt")))
    if datestr:
        fs = [f for f in fs if datestr in Path(f).name] or fs
    return Path(fs[-1]) if fs else None


def latest_morning_predictions(datestr=None):
    fs = sorted(glob.glob(str(DATA / "predictions_morning_*.json")))
    if datestr:
        fs = [f for f in fs if datestr in Path(f).name] or fs
    return Path(fs[-1]) if fs else None


# ---------- 解析 ----------

def parse_signal_section(rep_txt):
    """从早盘报告解析偏多/偏空名单与计数。"""
    out = {"adv": [], "dec": [], "adv_n": 0, "dec_n": 0}
    # 分段：偏多信号 (N只):  下面若干行  "  1. 名称 (代码) - 评分:4.9"
    for key, tag in (("adv", "偏多信号"), ("dec", "偏空信号")):
        m = re.search(rf"{tag}\s*\((\d+)只\)\s*:\s*\n(.*?)(?=\n\s*\n|\n[^\s\d]|\Z)", rep_txt, re.S)
        if not m:
            m2 = re.search(rf"{tag}\s*\((\d+)只\)", rep_txt)
            if m2:
                out[f"{key}_n"] = int(m2.group(1))
            continue
        out[f"{key}_n"] = int(m.group(1))
        for line in m.group(2).splitlines():
            mm = re.match(r"\s*\d+\.\s*(\S+?)\s*[（(](\d{6})[)）]", line)
            if mm:
                out[key].append((mm.group(1), mm.group(2)))
    return out


def parse_market_section(rep_txt):
    """从早盘报告解析市场情绪/状态。"""
    sent = ""
    state = ""
    m = re.search(r"市场情绪[:：]\s*(.+)", rep_txt)
    if m:
        sent = m.group(1).strip()
    m = re.search(r"市场状态[:：]\s*(.+)", rep_txt)
    if m:
        state = m.group(1).strip()
    return sent, state


def fallback_from_predictions():
    """报告缺失时，从预测 JSON 兜底算名单与计数。"""
    f = latest_morning_predictions(_today())
    if not f:
        return None
    try:
        p = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None
    preds = p.get("predictions") or []
    from collections import Counter
    c = Counter(x.get("signal") for x in preds)
    adv_n = c.get("买入", 0) + c.get("强烈买入", 0)
    dec_n = c.get("卖出", 0)
    adv = [(x["stock"]["name"], x["stock"]["symbol"]) for x in preds if x.get("signal") in ("买入", "强烈买入")]
    dec = [(x["stock"]["name"], x["stock"]["symbol"]) for x in preds if x.get("signal") == "卖出"]
    adv.sort(key=lambda t: t[1])
    dec.sort(key=lambda t: t[1])
    return {"adv": adv, "dec": dec, "adv_n": adv_n, "dec_n": dec_n, "total": len(preds)}


# ---------- 组装 ----------

def build_morning():
    datestr = _today()
    rep = latest_morning_report(datestr)
    if rep:
        rep_txt = rep.read_text(encoding="utf-8")
        sig = parse_signal_section(rep_txt)
        sent, state = parse_market_section(rep_txt)
    else:
        sig = fallback_from_predictions() or {"adv": [], "dec": [], "adv_n": 0, "dec_n": 0}
        sent, state = "", ""

    acc = compute_accuracy()
    cum = f"{acc['cum_hit'] * 100:.1f}%" if acc["cum_hit"] is not None else "—"
    ztxt = f"，z={acc['cum_z']:+.2f}" if acc["cum_z"] is not None else ""

    names = _fmt_names((sig["adv"] + sig["dec"])[:TOPN])
    market = " ".join(x for x in (sent, state) if x) or "数据暂不可用"

    lines = [
        f"📊 A股早盘 · {_cn_date(datestr)}",
        "",
        f"〔预测·待验证〕偏多{sig['adv_n']}/偏空{sig['dec_n']}：{names}",
        "〔配置框架·休眠〕ETF估值倾斜(股/债再平衡+沪深300估值z)已就绪，待你建仓后每日自动出份额级调仓单；当前无持仓→不操作",
        f"〔市场〕{market}",
        "",
        "——",
        f"⚠️ 预测为公式信号，累计方向一致率 {cum}{ztxt}，未过准确率门槛，仅供观察研究；准确率高后你再决定是否跟。{DISCLAIMER}",
    ]
    return datestr, "\n".join(lines)


def build_close():
    datestr = _today()
    rep = latest_morning_report(datestr)
    if rep:
        sig = parse_signal_section(rep.read_text(encoding="utf-8"))
    else:
        sig = fallback_from_predictions() or {"adv": [], "dec": [], "adv_n": 0, "dec_n": 0}

    names = _fmt_names((sig["adv"] + sig["dec"])[:TOPN])

    rep_txt = ""
    dr = sorted(glob.glob(str(REPORTS / "day_review_report_*.txt")))
    if dr:
        rep_txt = Path(dr[-1]).read_text(encoding="utf-8")

    hud = build_hud_lines(rep_txt)
    acc = compute_accuracy()
    if acc["cum_z"] is not None and acc["cum_z"] <= -2:
        review = "未过股神淘汰线（累计命中率显著差于硬币），继续观察不动"
    elif str(acc["cum_hit"]) != "None" and acc["cum_hit"] >= 0.55 and (acc["cum_z"] or 0) >= 2:
        review = "已过线，可关注跟单"
    else:
        review = "未过股神淘汰线，继续观察不动"

    lines = [
        f"📊 A股收盘 · {_cn_date(datestr)}",
        "",
        f"〔预测·当日〕偏多{sig['adv_n']}/偏空{sig['dec_n']}：{names}",
        "〔准确率看板〕",
    ] + hud + [
        f"〔复盘〕{review}",
        "",
        "——",
        f"⚠️ 预测为公式信号，当前未过准确率门槛，仅供观察研究；{DISCLAIMER}",
    ]
    return datestr, "\n".join(lines)


def main():
    mode = (sys.argv[1] if len(sys.argv) > 1 else "close").lower()
    if mode not in ("morning", "close"):
        print("usage: push_builder.py [morning|close]", file=sys.stderr)
        sys.exit(2)
    datestr, text = build_morning() if mode == "morning" else build_close()
    out = REPORTS / f"push_{mode}_{datestr}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[saved] {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
