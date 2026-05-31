"""
基金涨幅预测系统 — Streamlit Web 应用

⚠️ 免责声明：本程序仅为学习演示，所有预测结果均不构成投资建议。
基金投资有风险，过往业绩不代表未来表现，请谨慎决策。
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# 页面配置
st.set_page_config(
    page_title="基金涨幅预测系统",
    page_icon="📈",
    layout="wide",
)

# 导入自定义模块
from data_fetcher import fetch_fund_nav, get_fund_name, clear_cache
from indicators import compute_all, generate_signals, signal_summary
from predictor import train_and_predict

# ============================================================
# 侧边栏 — 参数输入
# ============================================================

st.sidebar.title("📈 基金涨幅预测系统")

# 风险提示
st.sidebar.warning(
    "⚠️ **风险提示**\n\n"
    "本程序仅为**学习演示**，所有预测结果均**不构成投资建议**。\n"
    "基金投资有风险，过往业绩不代表未来表现，请谨慎决策。"
)

st.sidebar.markdown("---")

code = st.sidebar.text_input(
    "基金代码",
    value="000001",
    placeholder="如: 000001（华夏成长）, 510050（上证50ETF）",
    help="输入6位基金代码，支持公募基金和ETF",
).strip()

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "起始日期",
        value=pd.Timestamp.now() - pd.DateOffset(years=2),
        help="数据查询起点",
    )
with col2:
    end_date = st.date_input(
        "截止日期",
        value=pd.Timestamp.now(),
        help="数据查询终点",
    )

pred_days = st.sidebar.slider(
    "预测天数",
    min_value=3,
    max_value=15,
    value=5,
    help="LSTM 预测未来多少个交易日的净值",
)

col_a, col_b = st.sidebar.columns(2)
with col_a:
    analyze_btn = st.sidebar.button("🚀 开始分析", type="primary", use_container_width=True)
with col_b:
    refresh_btn = st.sidebar.button("🔄 强制刷新", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "数据来源: akshare（东方财富等公开接口）\n"
    "模型: LSTM 神经网络 + XGBoost"
)

# ============================================================
# 主页面
# ============================================================

st.title("📈 基金涨幅预测系统")
st.caption("基于技术指标 + 机器学习（LSTM / XGBoost）的基金走势分析与趋势预测")

# 顶部风险横幅
st.warning(
    "⚠️ **重要声明**：本系统所有分析和预测结果仅供学习参考，**不构成任何投资建议**。"
    "基金投资存在亏损风险，请根据自身风险承受能力独立决策。"
    "过往业绩不代表未来表现。"
)

if not analyze_btn:
    st.info("👈 请在左侧输入基金代码并点击「开始分析」按钮")
    st.stop()

# ============================================================
# 数据加载
# ============================================================

with st.spinner(f"正在获取基金 {code} 的数据..."):
    try:
        if refresh_btn:
            clear_cache(code)

        fund_name = get_fund_name(code)
        df_raw = fetch_fund_nav(
            code,
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
            force_refresh=refresh_btn,
        )

        if df_raw.empty:
            st.error(f"未获取到基金 {code} 的数据，请检查代码是否正确")
            st.stop()

        st.success(f"✅ 已加载 **{fund_name}**（{code}），共 {len(df_raw)} 条数据")

    except Exception as e:
        st.error(f"数据获取失败: {e}")
        st.stop()

# 计算指标
with st.spinner("正在计算技术指标..."):
    df = compute_all(df_raw)
    df_signal = generate_signals(df)

# ============================================================
# Tab 布局
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 历史走势", "📉 技术指标", "🔔 买卖信号", "🤖 AI 预测"]
)

# ============================================================
# Tab 1: 历史走势
# ============================================================

with tab1:
    st.subheader(f"{fund_name}（{code}）历史净值走势")

    fig1 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("净值 & 均线", "日涨跌幅 (%)"),
    )

    # 主图：净值 + 均线
    fig1.add_trace(
        go.Scatter(x=df["date"], y=df["nav"], mode="lines", name="净值", line=dict(color="#1f77b4", width=2)),
        row=1, col=1,
    )
    colors = {"MA5": "#ff7f0e", "MA10": "#2ca02c", "MA20": "#d62728", "MA60": "#9467bd"}
    for ma, color in colors.items():
        fig1.add_trace(
            go.Scatter(x=df["date"], y=df[ma], mode="lines", name=ma, line=dict(color=color, width=1, dash="dot")),
            row=1, col=1,
        )

    # 副图：涨跌幅柱状图
    colors_bar = ["#ef5350" if v < 0 else "#26a69a" for v in df["change_pct"].fillna(0)]
    fig1.add_trace(
        go.Bar(x=df["date"], y=df["change_pct"], name="日涨跌幅%", marker_color=colors_bar, opacity=0.7),
        row=2, col=1,
    )

    fig1.update_layout(height=600, hovermode="x unified", showlegend=True)
    fig1.update_yaxes(title_text="净值", row=1, col=1)
    fig1.update_yaxes(title_text="涨跌幅 %", row=2, col=1)

    st.plotly_chart(fig1, use_container_width=True)

    # 统计信息
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("最新净值", f"{df['nav'].iloc[-1]:.4f}")
    with col_b:
        st.metric("近期最高", f"{df['nav'].max():.4f}")
    with col_c:
        st.metric("近期最低", f"{df['nav'].min():.4f}")
    with col_d:
        total_change = (df["nav"].iloc[-1] - df["nav"].iloc[0]) / df["nav"].iloc[0] * 100
        st.metric("区间涨跌", f"{total_change:+.2f}%")

# ============================================================
# Tab 2: 技术指标
# ============================================================

with tab2:
    st.subheader("技术指标分析")

    # MACD
    st.markdown("#### MACD（趋势指标）")
    fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.4])

    fig_macd.add_trace(
        go.Scatter(x=df["date"], y=df["nav"], mode="lines", name="净值", line=dict(color="#1f77b4")),
        row=1, col=1,
    )
    fig_macd.add_trace(
        go.Scatter(x=df["date"], y=df["MACD"], mode="lines", name="MACD", line=dict(color="#1f77b4")),
        row=2, col=1,
    )
    fig_macd.add_trace(
        go.Scatter(x=df["date"], y=df["MACD_signal"], mode="lines", name="Signal", line=dict(color="#ff7f0e")),
        row=2, col=1,
    )
    # 柱状图
    colors_hist = ["#ef5350" if v < 0 else "#26a69a" for v in df["MACD_hist"].fillna(0)]
    fig_macd.add_trace(
        go.Bar(x=df["date"], y=df["MACD_hist"], name="Histogram", marker_color=colors_hist, opacity=0.5),
        row=2, col=1,
    )
    fig_macd.update_layout(height=450, hovermode="x unified")
    st.plotly_chart(fig_macd, use_container_width=True)

    # RSI
    st.markdown("#### RSI（相对强弱指标）")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(
        go.Scatter(x=df["date"], y=df["RSI"], mode="lines", name="RSI(14)", line=dict(color="#9467bd", width=2))
    )
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超买 70")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超卖 30")
    fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray")
    fig_rsi.update_layout(height=300, hovermode="x unified", yaxis_range=[0, 100])
    st.plotly_chart(fig_rsi, use_container_width=True)

    # 布林带
    st.markdown("#### 布林带（Bollinger Bands）")
    fig_bb = go.Figure()
    fig_bb.add_trace(
        go.Scatter(x=df["date"], y=df["BB_upper"], mode="lines", name="上轨", line=dict(color="gray", width=1, dash="dash"))
    )
    fig_bb.add_trace(
        go.Scatter(x=df["date"], y=df["BB_middle"], mode="lines", name="中轨(MA20)", line=dict(color="orange", width=1))
    )
    fig_bb.add_trace(
        go.Scatter(x=df["date"], y=df["BB_lower"], mode="lines", name="下轨", line=dict(color="gray", width=1, dash="dash"),
                   fill="tonexty", fillcolor="rgba(128,128,128,0.1)")
    )
    fig_bb.add_trace(
        go.Scatter(x=df["date"], y=df["nav"], mode="lines", name="净值", line=dict(color="#1f77b4", width=2))
    )
    fig_bb.update_layout(height=400, hovermode="x unified")
    st.plotly_chart(fig_bb, use_container_width=True)

# ============================================================
# Tab 3: 买卖信号
# ============================================================

with tab3:
    st.subheader("交易信号检测")

    summary = signal_summary(df_signal)

    # 当前状态卡片
    st.markdown("### 📋 当前状态")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("日期", summary["latest_date"])
    with col_b:
        st.metric("最新净值", f"{summary['latest_nav']:.4f}")
    with col_c:
        st.metric("趋势判断", summary["trend"])
    with col_d:
        st.metric("RSI 状态", summary["rsi_status"])

    st.markdown("---")

    # 信号图表
    st.markdown("### 📈 买卖信号标注图")
    fig_sig = go.Figure()
    fig_sig.add_trace(
        go.Scatter(x=df_signal["date"], y=df_signal["nav"], mode="lines", name="净值", line=dict(color="#1f77b4"))
    )

    # 标注买入信号
    buy_df = df_signal[df_signal["signal"] == 1]
    if not buy_df.empty:
        fig_sig.add_trace(
            go.Scatter(
                x=buy_df["date"], y=buy_df["nav"],
                mode="markers", name="买入信号",
                marker=dict(symbol="triangle-up", size=12, color="#26a69a", line=dict(width=1, color="darkgreen")),
                text=buy_df["signal_reason"], hoverinfo="text+x+y",
            )
        )

    # 标注卖出信号
    sell_df = df_signal[df_signal["signal"] == -1]
    if not sell_df.empty:
        fig_sig.add_trace(
            go.Scatter(
                x=sell_df["date"], y=sell_df["nav"],
                mode="markers", name="卖出信号",
                marker=dict(symbol="triangle-down", size=12, color="#ef5350", line=dict(width=1, color="darkred")),
                text=sell_df["signal_reason"], hoverinfo="text+x+y",
            )
        )

    fig_sig.update_layout(height=500, hovermode="closest")
    st.plotly_chart(fig_sig, use_container_width=True)

    # 近期信号表
    st.markdown("### 📝 近期信号记录")
    if summary["recent_signals"]:
        sig_df = pd.DataFrame(summary["recent_signals"])
        sig_df = sig_df[::-1]  # 最新的在前
        st.dataframe(sig_df, use_container_width=True)
    else:
        st.info("近期无明确买卖信号")

# ============================================================
# Tab 4: AI 预测
# ============================================================

with tab4:
    st.subheader("🤖 AI 模型预测")

    st.warning(
        "⚠️ **预测结果仅供学习参考，绝不构成投资建议。** "
        "机器学习模型基于历史数据拟合，无法预测突发事件、政策变化等不可预见因素。"
    )

    with st.spinner("正在训练模型并预测...（首次运行可能需要数十秒）"):
        try:
            result = train_and_predict(df, code, pred_days=pred_days)
        except Exception as e:
            st.error(f"模型训练/预测失败: {e}")
            st.info("可能原因：数据量太少（建议至少半年以上数据）、数据质量问题等。请尝试扩大日期范围。")
            st.stop()

    # ---- 预测结果卡片 ----
    st.markdown("### 📊 预测结果概览")

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("当前净值", f"{result['last_nav']:.4f}")
    with col_b:
        pred_last = result["lstm_pred"][-1]
        delta = result["pred_change_pct"]
        st.metric(
            f"{pred_days}日后预测净值",
            f"{pred_last:.4f}",
            delta=f"{delta:+.2f}%",
        )
    with col_c:
        trend_emoji = "📈" if result["lstm_trend"] == "看涨" else ("📉" if result["lstm_trend"] == "看跌" else "📊")
        st.metric("LSTM 趋势", f"{trend_emoji} {result['lstm_trend']}")
    with col_d:
        up_pct = result["xgb_up_prob"] * 100
        st.metric("次日上涨概率（XGBoost）", f"{up_pct:.1f}%")

    # ---- LSTM 预测图 ----
    st.markdown("### 🔮 LSTM 净值预测")

    # 组合历史最后60天 + 预测
    hist_part = df.tail(60)
    pred_dates = pd.to_datetime(result["lstm_dates"])
    pred_vals = result["lstm_pred"]

    fig_pred = go.Figure()

    # 历史部分
    fig_pred.add_trace(
        go.Scatter(
            x=hist_part["date"], y=hist_part["nav"],
            mode="lines", name="历史净值",
            line=dict(color="#1f77b4", width=2),
        )
    )

    # 预测部分（虚线 + 置信区间背景）
    last_hist_val = hist_part["nav"].iloc[-1]
    all_pred_vals = [last_hist_val] + list(pred_vals)
    all_pred_dates = [hist_part["date"].iloc[-1]] + list(pred_dates)

    fig_pred.add_trace(
        go.Scatter(
            x=all_pred_dates, y=all_pred_vals,
            mode="lines+markers", name="LSTM 预测",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
            marker=dict(size=8, symbol="circle"),
        )
    )

    # 预测区间（简单 ± 标准差估计）
    std_est = hist_part["change_pct"].std() / 100 * np.array(all_pred_vals)
    upper = np.array(all_pred_vals) + std_est * 2
    lower = np.array(all_pred_vals) - std_est * 2

    fig_pred.add_trace(
        go.Scatter(
            x=list(all_pred_dates) + list(all_pred_dates)[::-1],
            y=list(upper) + list(lower)[::-1],
            fill="toself", fillcolor="rgba(255,127,14,0.15)",
            line=dict(color="rgba(255,127,14,0)"),
            name="置信区间 (±2σ)",
            showlegend=True,
        )
    )

    # 虚线圈出预测区间
    fig_pred.add_vline(
        x=hist_part["date"].iloc[-1],
        line_dash="dot", line_color="gray",
        annotation_text="← 历史 | 预测 →",
    )

    fig_pred.update_layout(height=450, hovermode="x unified")
    st.plotly_chart(fig_pred, use_container_width=True)

    # ---- 预测详情表 ----
    st.markdown("### 📋 逐日预测明细")
    pred_table = pd.DataFrame(
        {
            "日期": result["lstm_dates"],
            "预测净值": result["lstm_pred"],
            "较前日变化": [0] + [
                round(result["lstm_pred"][i] - result["lstm_pred"][i - 1], 4)
                for i in range(1, len(result["lstm_pred"]))
            ],
            "累计变化%": [
                round((v - result["last_nav"]) / result["last_nav"] * 100, 2)
                for v in result["lstm_pred"]
            ],
        }
    )
    st.dataframe(pred_table, use_container_width=True)

    # ---- 综合研判 ----
    st.markdown("### 📝 综合研判")

    lstm_trend = result["lstm_trend"]
    xgb_prob = result["xgb_up_prob"]
    confidence = result["confidence_note"]

    # 多空信号综合
    if lstm_trend == "看涨" and xgb_prob > 0.5:
        overall = "偏多"
        detail = f"LSTM 预测{lstm_trend}，XGBoost 也显示上涨概率 {xgb_prob*100:.1f}%，两者方向一致。"
    elif lstm_trend == "看跌" and xgb_prob < 0.5:
        overall = "偏空"
        detail = f"LSTM 预测{lstm_trend}，XGBoost 上涨概率仅 {xgb_prob*100:.1f}%，两者方向一致。"
    else:
        overall = "分歧"
        detail = f"LSTM 趋势为「{lstm_trend}」，但 XGBoost 次日上涨概率为 {xgb_prob*100:.1f}%，模型间存在分歧，建议观望。"

    st.info(
        f"**综合研判：{overall}**（置信度：{confidence}）\n\n"
        f"{detail}\n\n"
        f"⚠️ 再次提醒：以上分析完全基于历史数据的统计规律和机器学习拟合，"
        f"**不能预测政策变化、市场情绪、突发事件**等实际影响基金走势的关键因素。"
        f"请将此结果仅作为学习参考，实际投资请咨询专业顾问。"
    )

# ============================================================
# 页脚
# ============================================================

st.markdown("---")
st.caption(
    "📈 基金涨幅预测系统 | 数据来源: akshare | "
    "技术指标: pandas-ta | ML: PyTorch LSTM + XGBoost | "
    "⚠️ 仅供学习参考，不构成投资建议"
)
