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
import statistics as st
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prediction_accuracy_hud import build_hud_lines, compute_accuracy  # noqa: E402
import portfolio_manager as pm  # noqa: E402

TOPN = 8
DISCLAIMER = "本推送为系统自动生成的公式信号/研究记录，不构成个人投资建议。"

# ---------- 估值倾斜配置信号（系统唯一经统计验证的可操作引擎） ----------
# 股票腿 = 沪深300(sh000300, ETF 510300)；债券腿 = 上证国债(sh000012, ETF 511010)
# 基准 60/40，年度再平衡；目标股比 = clamp(0.60 - 0.15*z, 0.40, 0.80)
# z = 沪深300 价格相对 5 年均线(1220交易日) 的 z-score
# 已预注册重验：t_NW=+2.73（全样本显著），回撤 -28.4%（vs 固定 -42.6%）
CACHE_DIR = DATA / "style_index_cache_6000"
MA_WIN = 1220
BASE = 0.60
LO, HI = 0.40, 0.80
K = 0.15
TILT_T = 2.73  # 该引擎的 Newey-West t 值（显著性锚）


def _load_or_fetch(sym, cache_f, refresh):
    """优先用缓存（末日期够新直接用），否则拉 sina 日线；失败回退旧缓存。"""
    if cache_f.exists() and not refresh:
        try:
            d = json.loads(cache_f.read_text(encoding="utf-8"))
            if d and (datetime.now() - datetime.strptime(d[-1][0], "%Y-%m-%d")).days < 10:
                return dict(d)
        except Exception:
            pass
    try:
        u = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
             f"CN_MarketData.getKLineData?symbol={sym}&scale=240&ma=no&datalen=6000")
        raw = urllib.request.urlopen(urllib.request.Request(
            u, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"}),
            timeout=25).read().decode()
        arr = json.loads(raw)
        out = [(r["day"], float(r["close"])) for r in arr if r.get("close")]
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_f.write_text(json.dumps(out), encoding="utf-8")
        return dict(out)
    except Exception:
        if cache_f.exists():
            try:
                return dict(json.loads(cache_f.read_text(encoding="utf-8")))
            except Exception:
                return None
        return None


def compute_tilt_signal(refresh=False):
    """算估值倾斜实时信号。返回 dict 或 None（行情失败）。"""
    stock = _load_or_fetch("sh000300", CACHE_DIR / "sh000300_6000.json", refresh)
    bond = _load_or_fetch("sh000012", CACHE_DIR / "sh000012_6000.json", refresh)
    if not stock or not bond:
        return None
    dates = sorted(d for d in stock if d in bond)
    prices = [stock[d] for d in dates]
    i = len(dates) - 1
    if i - MA_WIN < 0:
        return None
    win = prices[i - MA_WIN:i + 1]
    mu, sd = st.mean(win), st.pstdev(win)
    if sd == 0:
        return None
    z = (prices[i] - mu) / sd
    tgt = min(HI, max(LO, BASE - K * z))
    return {
        "z": z,
        "target_equity": tgt,
        "target_bond": 1 - tgt,
        "last_date": dates[-1],
        "last_close": prices[-1],
    }


def tilt_signal_line(tilt):
    """单行可读的倾斜信号描述。"""
    if tilt is None:
        return "行情获取失败，信号暂不可用"
    z = tilt["z"]
    label = "偏贵" if z > 0.3 else ("偏便宜" if z < -0.3 else "中性")
    eq = tilt["target_equity"] * 100
    bd = tilt["target_bond"] * 100
    return (f"沪深300 z={z:+.2f}（{label}）→ 目标 股{eq:.1f}% / 债{bd:.1f}%"
            f"（收盘 {tilt['last_close']:.2f} @ {tilt['last_date']}）")


def rebalance_demo_lines():
    """若 portfolio.json 有持仓，生成份额级再平衡指令（虚拟则标注演示）。

    复用 portfolio_manager 的 load/evaluate/actions（只读、无写文件副作用）。
    取价失败（无网络）或无持仓 → 返回 []，由调用方回退到比例级建仓指令。
    """
    pf = pm.load()
    if not pf or not pf.get("holdings"):
        return []
    try:
        ev = pm.evaluate(pf)
    except Exception:
        return []
    if not ev.get("ok"):
        return []
    acts = pm.actions(pf)
    pick = [l for l in acts if l.startswith(("🔧", "✅", "当前"))]
    if not pick:
        return []
    # 剥掉交易所前缀 sh/sz，显示纯6位代码（如 510300 / 511010）
    clean = [re.sub(r"\b(?:sh|sz)(\d{6})\b", r"\1", l) for l in pick]
    if pf.get("virtual"):
        tag = f"（虚拟持仓 {ev['total']:,.0f}元 · 非真实资金演示）"
    else:
        tag = "（基于你的真实持仓）"
    return [f"〔份额级调仓·演示〕{tag}"] + ["  " + l for l in clean]


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
        sent, state = "", ""

    tilt = compute_tilt_signal()
    tl = tilt_signal_line(tilt)
    market = " ".join(x for x in (sent, state) if x) or "数据暂不可用"
    demo = rebalance_demo_lines()

    lines = [
        f"📊 A股早盘 · {_cn_date(datestr)}",
        "",
        f"〔配置信号·唯一可操作〕{tl}",
    ]
    if demo:
        lines += demo
    else:
        eq = tilt["target_equity"] * 100 if tilt else 0
        bd = tilt["target_bond"] * 100 if tilt else 0
        lines.append(f"  建仓指令（当前无持仓）：按比例买入 510300(沪深300ETF) {eq:.1f}% + 511010(国债ETF) {bd:.1f}%（用你的总资金）")
    lines += [
        "  偏离>5pp 或满一年自动再平衡，建仓后每日推送份额级调仓单",
        f"〔市场〕{market}",
        "",
        "——",
        f"⚠️ 个股日频预测线已关闭（1006样本累计方向一致率42.9%、z=−4.48，显著低于硬币，无选股edge）。"
        f"系统唯一经统计验证的买卖引擎=估值倾斜配置（t_NW=+{TILT_T}，回撤−28.4%有界）。{DISCLAIMER}",
    ]
    return datestr, "\n".join(lines)


def build_close():
    datestr = _today()
    tilt = compute_tilt_signal()
    tl = tilt_signal_line(tilt)
    demo = rebalance_demo_lines()

    rep_txt = ""
    dr = sorted(glob.glob(str(REPORTS / "day_review_report_*.txt")))
    if dr:
        rep_txt = Path(dr[-1]).read_text(encoding="utf-8")

    hud = build_hud_lines(rep_txt)

    lines = [
        f"📊 A股收盘 · {_cn_date(datestr)}",
        "",
        f"〔配置信号·唯一可操作〕{tl}",
    ]
    if demo:
        lines += demo
    else:
        lines.append("  偏离未达触发线→今日无需操作（已建仓者）；未建仓者按上方目标比例建仓")
    lines += [
        "〔已关闭线·诚实存档〕日频个股预测 累计一致率42.9% n=1006 z=−4.48 p=7.6e-6（显著反向，已停推）",
        "〔准确率看板·历史证据〕",
    ] + hud + [
        "",
        "——",
        f"⚠️ 个股日频预测线已关闭，无选股edge；唯一可操作引擎=估值倾斜配置（t_NW=+{TILT_T}）。{DISCLAIMER}",
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
