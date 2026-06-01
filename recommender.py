"""
每日基金推荐引擎 — 扫描精选基金池，结合技术指标 + ML 预测 + AI 分析，
给出当日最值得关注的基金推荐。
"""

import json
import os
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from data_fetcher import fetch_fund_nav, get_fund_name
from indicators import compute_all
from llm_analyzer import _ollama_generate, check_ollama_available

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
REC_CACHE_FILE = os.path.join(CACHE_DIR, "daily_rec.json")

# ============================================================
# 精选基金池 — 覆盖不同类别
# ============================================================

WATCHLIST = [
    # ---- 宽基指数 ETF ----
    {"code": "510050", "category": "宽基ETF", "desc": "上证50ETF"},
    {"code": "510300", "category": "宽基ETF", "desc": "沪深300ETF"},
    {"code": "510500", "category": "宽基ETF", "desc": "中证500ETF"},
    {"code": "159915", "category": "宽基ETF", "desc": "创业板ETF"},
    {"code": "588000", "category": "宽基ETF", "desc": "科创50ETF"},

    # ---- 行业/主题 ETF ----
    {"code": "512880", "category": "行业ETF", "desc": "证券ETF"},
    {"code": "512690", "category": "行业ETF", "desc": "酒ETF"},
    {"code": "516510", "category": "行业ETF", "desc": "芯片ETF"},
    {"code": "159766", "category": "行业ETF", "desc": "新能源车ETF"},
    {"code": "512010", "category": "行业ETF", "desc": "医药ETF"},

    # ---- 优秀主动基金 ----
    {"code": "161725", "category": "主动基金", "desc": "招商中证白酒"},
    {"code": "005827", "category": "主动基金", "desc": "易方达蓝筹精选"},
    {"code": "002190", "category": "主动基金", "desc": "农银新能源主题"},
    {"code": "320007", "category": "主动基金", "desc": "诺安成长混合"},
    {"code": "005919", "category": "主动基金", "desc": "天弘沪深300ETF联接C"},
]


def _load_rec_cache() -> dict | None:
    """加载每日推荐缓存（当天有效）。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.exists(REC_CACHE_FILE):
        return None
    try:
        with open(REC_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        cache_date = data.get("date", "")
        today = datetime.now().strftime("%Y-%m-%d")
        if cache_date == today:
            return data
    except Exception:
        pass
    return None


def _save_rec_cache(data: dict):
    """保存每日推荐缓存。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(REC_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, default=str)


# ============================================================
# 单基金评分
# ============================================================

def score_fund(df: pd.DataFrame, pred_result: dict | None = None) -> dict:
    """
    对单只基金进行多维度评分（满分 100）。

    返回:
        {
            "total_score": float,      # 0-100
            "tech_score": float,       # 技术面得分 0-40
            "pred_score": float,       # ML预测得分 0-40
            "perf_score": float,       # 近期表现得分 0-20
            "details": list[str],      # 关键指标描述
            "badge": str,              # "推荐" / "关注" / "观望"
        }
    """
    details = []

    # ---- 1. 技术面评分 (40分) ----
    tech_score = 20.0  # 基础分

    last_rsi = df["RSI"].iloc[-1] if "RSI" in df.columns else 50
    last_macd_hist = df["MACD_hist"].iloc[-1] if "MACD_hist" in df.columns else 0
    last_ma5 = df["MA5"].iloc[-1] if "MA5" in df.columns else 0
    last_ma20 = df["MA20"].iloc[-1] if "MA20" in df.columns else 0
    last_nav = df["nav"].iloc[-1]
    bb_lower = df["BB_lower"].iloc[-1] if "BB_lower" in df.columns else 0
    bb_upper = df["BB_upper"].iloc[-1] if "BB_upper" in df.columns else 0

    # RSI: 30-70 区间最佳，接近 30 超卖区加分多
    if not pd.isna(last_rsi):
        if 30 <= last_rsi <= 70:
            tech_score += 8
            details.append(f"RSI={last_rsi:.1f} (正常区间)")
        elif last_rsi < 30:
            tech_score += 12  # 超卖 = 潜在买入机会
            details.append(f"RSI={last_rsi:.1f} (超卖区 💡)")
        else:
            tech_score += 2
            details.append(f"RSI={last_rsi:.1f} (超买区 ⚠️)")

    # MACD: 金叉状态加分
    if not pd.isna(last_macd_hist):
        if last_macd_hist > 0:
            # MACD 柱在扩大还是缩小
            prev_hist = df["MACD_hist"].iloc[-2] if len(df) >= 2 else 0
            if last_macd_hist > prev_hist:
                tech_score += 8
                details.append("MACD 红柱放大 (多头增强)")
            else:
                tech_score += 5
                details.append("MACD 红柱缩小 (多头减弱)")
        else:
            tech_score += 1
            details.append("MACD 绿柱 (空头)")

    # 均线排列: MA5 > MA20 加分
    if last_ma5 > last_ma20:
        tech_score += 6
        details.append("MA5 > MA20 (短期多头)")
    else:
        tech_score += 1
        details.append("MA5 < MA20 (短期空头)")

    # 布林带位置
    if not pd.isna(bb_lower) and not pd.isna(bb_upper):
        if last_nav <= bb_lower * 1.02:
            tech_score += 6
            details.append("接近布林下轨 (超卖)")
        elif last_nav >= bb_upper * 0.98:
            tech_score += 0
            details.append("接近布林上轨 (超买)")
        else:
            tech_score += 4
            details.append("位于布林带中轨附近")

    tech_score = min(40, max(0, tech_score))

    # ---- 2. ML 预测评分 (40分) ----
    pred_score = 20.0  # 基础分

    if pred_result:
        lstm_trend = pred_result.get("lstm_trend", "震荡")
        xgb_prob = pred_result.get("xgb_up_prob", 0.5)
        pred_change = pred_result.get("pred_change_pct", 0)
        confidence = pred_result.get("confidence_note", "")

        # LSTM 趋势
        if lstm_trend == "看涨":
            pred_score += 8
            details.append(f"LSTM 预测看涨 (+{pred_change}%)")
        elif lstm_trend == "看跌":
            pred_score += 2
            details.append(f"LSTM 预测看跌 ({pred_change}%)")
        else:
            pred_score += 5
            details.append("LSTM 预测震荡")

        # XGBoost 上涨概率
        if xgb_prob > 0.65:
            pred_score += 10
            details.append(f"XGBoost 上涨概率 {xgb_prob*100:.0f}% (高)")
        elif xgb_prob > 0.5:
            pred_score += 6
            details.append(f"XGBoost 上涨概率 {xgb_prob*100:.0f}% (中等)")
        elif xgb_prob > 0.35:
            pred_score += 3
            details.append(f"XGBoost 上涨概率 {xgb_prob*100:.0f}% (偏弱)")
        else:
            pred_score += 1
            details.append(f"XGBoost 上涨概率 {xgb_prob*100:.0f}% (低)")

        # 置信度加分
        if "较高" in confidence:
            pred_score += 2
    else:
        pred_score += 5  # 无预测时中等分
        details.append("ML 预测不可用")

    pred_score = min(40, max(0, pred_score))

    # ---- 3. 近期表现 (20分) ----
    perf_score = 10.0

    recent_5 = df["change_pct"].tail(5)
    if len(recent_5) >= 3:
        avg_change = recent_5.mean()
        volatility = recent_5.std()

        # 涨幅（温和上涨最好）
        if 0.1 < avg_change < 1.0:
            perf_score += 6
            details.append(f"近5日均涨 {avg_change:+.2f}% (温和上涨)")
        elif avg_change >= 1.0:
            perf_score += 3
            details.append(f"近5日均涨 {avg_change:+.2f}% (大涨,追高风险)")
        elif avg_change >= -0.3:
            perf_score += 4
            details.append(f"近5日均涨 {avg_change:+.2f}% (小幅震荡)")
        else:
            perf_score += 1
            details.append(f"近5日均涨 {avg_change:+.2f}% (明显下跌)")

        # 波动率（低波动加分）
        if volatility < 0.3:
            perf_score += 4
            details.append(f"波动率低 ({volatility:.2f}%)")
        elif volatility < 0.8:
            perf_score += 2
            details.append(f"波动率中等 ({volatility:.2f}%)")
        else:
            details.append(f"波动率高 ({volatility:.2f}%)")

    perf_score = min(20, max(0, perf_score))

    # ---- 综合 ----
    total = tech_score + pred_score + perf_score

    # 评级
    if total >= 75:
        badge = "🌟 强烈推荐"
    elif total >= 65:
        badge = "👍 推荐关注"
    elif total >= 50:
        badge = "👀 一般关注"
    else:
        badge = "⏸️ 暂时观望"

    return {
        "total_score": round(total, 1),
        "tech_score": round(tech_score, 1),
        "pred_score": round(pred_score, 1),
        "perf_score": round(perf_score, 1),
        "details": details,
        "badge": badge,
        "last_nav": round(float(last_nav), 4),
        "rsi": round(float(last_rsi), 1) if not pd.isna(last_rsi) else None,
        "macd_hist": round(float(last_macd_hist), 4) if not pd.isna(last_macd_hist) else None,
    }


# ============================================================
# 批量扫描
# ============================================================

def run_daily_scan(
    force_refresh: bool = False,
    progress_callback=None,
) -> list[dict]:
    """
    扫描基金池中的所有基金，返回评分排序结果。

    参数:
        force_refresh: 是否强制刷新（忽略缓存）
        progress_callback: 可选进度回调 fn(current, total, fund_name)

    返回:
        list[dict]: 按 total_score 降序排列的基金推荐列表
    """
    # 检查缓存
    if not force_refresh:
        cached = _load_rec_cache()
        if cached and "results" in cached:
            return cached["results"]

    results = []
    total = len(WATCHLIST)

    for i, fund in enumerate(WATCHLIST):
        code = fund["code"]
        name = f"{fund['desc']} ({code})"

        if progress_callback:
            progress_callback(i + 1, total, name)

        try:
            # 获取数据（开启缓存加速）
            df_raw = fetch_fund_nav(code, force_refresh=False)
            if df_raw.empty:
                results.append({
                    "code": code,
                    "name": fund["desc"],
                    "category": fund["category"],
                    "total_score": 0,
                    "badge": "❌ 数据缺失",
                    "details": ["无法获取净值数据"],
                    "error": True,
                })
                continue

            # 计算技术指标
            df = compute_all(df_raw)

            # 尝试 ML 预测
            pred_result = None
            try:
                from predictor import train_and_predict
                pred_result = train_and_predict(df, code, pred_days=5, force_retrain=False)
            except Exception:
                pass

            # 评分
            scores = score_fund(df, pred_result)

            results.append({
                "code": code,
                "name": fund["desc"],
                "full_name": get_fund_name(code),
                "category": fund["category"],
                **scores,
                "pred_result": pred_result,
                "error": False,
            })

        except Exception as e:
            results.append({
                "code": code,
                "name": fund["desc"],
                "category": fund["category"],
                "total_score": 0,
                "badge": "❌ 分析失败",
                "details": [str(e)[:100]],
                "error": True,
            })

    # 按评分降序排列
    results.sort(key=lambda x: x["total_score"], reverse=True)

    # 保存缓存
    _save_rec_cache({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "results": [{k: v for k, v in r.items() if k != "pred_result"} for r in results],
    })

    return results


# ============================================================
# Ollama AI 推荐理由
# ============================================================

def generate_recommendation(top_funds: list[dict], market_context: str = "") -> str:
    """
    使用 Ollama 生成每日推荐总结。

    参数:
        top_funds: 前 5-8 只基金
        market_context: 可选的市场概览文字

    返回:
        AI 生成的推荐总结文字
    """
    ok, _ = check_ollama_available()
    if not ok:
        return "[Ollama 未启动，无法生成 AI 推荐总结]"

    # 构建基金摘要
    fund_lines = []
    for i, f in enumerate(top_funds[:8]):
        if f.get("error"):
            continue
        name = f.get("name", f.get("code", "?"))
        code = f.get("code", "")
        score = f.get("total_score", 0)
        badge = f.get("badge", "")
        details = "; ".join(f.get("details", [])[:3])
        fund_lines.append(
            f"{i+1}. {name}（{code}）— 评分 {score}/100 [{badge}]\n"
            f"   指标: {details}"
        )

    if not fund_lines:
        return "当前无可分析的基金数据，请检查网络连接后重试。"

    funds_text = "\n".join(fund_lines)

    prompt = f"""你是一位资深基金投顾，请根据以下基金评分数据，生成一份简明的「今日基金关注推荐」总结。

【评分说明】
- 满分 100 分：技术面 40 + ML预测 40 + 近期表现 20
- 评分 ≥75：强烈推荐 | 65-74：推荐关注 | 50-64：一般关注 | <50：暂时观望

【今日基金排名】
{funds_text}

请生成 200-300 字的推荐总结，包含：
1. 今日市场关注要点（1-2句，基于数据推测）
2. Top 3 推荐基金及理由（每只1句）
3. 需要谨慎的基金（如果有的话，1句）
4. 结尾强调「仅供参考，不构成投资建议」

请用中文，口语化但不失专业。直接返回正文，不要输出 JSON 或其他格式。"""

    system = "你是资深基金投顾，专业、客观、谨慎。基于数据给出分析，不做确定性预测。"
    result = _ollama_generate(prompt, system, timeout=180)

    return result.strip()
