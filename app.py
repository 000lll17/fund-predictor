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
from report import generate_offline_report

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

    st.plotly_chart(fig1, width="stretch")

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
    st.plotly_chart(fig_macd, width="stretch")

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
    st.plotly_chart(fig_rsi, width="stretch")

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
    st.plotly_chart(fig_bb, width="stretch")

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
    st.plotly_chart(fig_sig, width="stretch")

    # 近期信号表
    st.markdown("### 📝 近期信号记录")
    if summary["recent_signals"]:
        sig_df = pd.DataFrame(summary["recent_signals"])
        sig_df = sig_df[::-1]  # 最新的在前
        st.dataframe(sig_df, width="stretch")
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

    # ---- 预测结果概览卡片 ----
    st.markdown("### 📊 预测结果概览")

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("📌 当前净值", f"{result['last_nav']:.4f}")
    with col_b:
        pred_last = result["lstm_pred"][-1]
        delta = result["pred_change_pct"]
        st.metric(
            f"🔮 {pred_days}日后预测",
            f"{pred_last:.4f}",
            delta=f"{delta:+.2f}%",
        )
    with col_c:
        trend_emoji = "📈" if result["lstm_trend"] == "看涨" else ("📉" if result["lstm_trend"] == "看跌" else "📊")
        st.metric("LSTM 趋势判断", f"{trend_emoji} {result['lstm_trend']}")
    with col_d:
        up_pct = result["xgb_up_prob"] * 100
        arrow = "🟢" if up_pct >= 50 else "🔴"
        st.metric("次日上涨概率", f"{arrow} {up_pct:.1f}%")

    # ---- 风险 & 波动指标 ----
    st.markdown("### 📉 风险 & 波动分析")
    hist_vol = df["change_pct"].std()
    recent_vol = df.tail(30)["change_pct"].std()
    max_drawdown = (df["nav"] / df["nav"].cummax() - 1).min() * 100
    pred_volatility = np.std(result["lstm_pred"]) / result["last_nav"] * 100

    # 计算支撑位和阻力位
    recent_high = df.tail(60)["nav"].max()
    recent_low = df.tail(60)["nav"].min()
    bb_upper_val = df["BB_upper"].iloc[-1]
    bb_lower_val = df["BB_lower"].iloc[-1]
    ma20_val = df["MA20"].iloc[-1]
    ma60_val = df["MA60"].iloc[-1]

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        st.metric("📊 历史波动率", f"{hist_vol:.2f}%",
                  delta=f"近30日 {recent_vol:.2f}%",
                  delta_color="off")
    with col_r2:
        st.metric("📉 最大回撤", f"{max_drawdown:.2f}%")
    with col_r3:
        st.metric("🎯 预测波动", f"{pred_volatility:.2f}%",
                  help="预测期内净值波动幅度，越大风险越高")
    with col_r4:
        risk_level = "低风险" if hist_vol < 1 else ("中风险" if hist_vol < 2.5 else "高风险")
        st.metric("⚠️ 风险等级", risk_level)

    # ---- 关键价位 ----
    st.markdown("### 🎯 关键价位参考")
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.metric("🔴 阻力位 (60日高)", f"{recent_high:.4f}",
                  delta=f"距当前 {((recent_high - result['last_nav']) / result['last_nav'] * 100):.2f}%")
    with col_k2:
        st.metric("🟢 支撑位 (60日低)", f"{recent_low:.4f}",
                  delta=f"距当前 {((recent_low - result['last_nav']) / result['last_nav'] * 100):.2f}%")
    with col_k3:
        support = bb_lower_val if not pd.isna(bb_lower_val) else ma60_val
        st.metric("📗 布林下轨", f"{support:.4f}",
                  help="价格跌破此线通常视为超卖")
    with col_k4:
        resist = bb_upper_val if not pd.isna(bb_upper_val) else recent_high
        st.metric("📕 布林上轨", f"{resist:.4f}",
                  help="价格突破此线通常视为超买")

    # ---- LSTM 预测图 ----
    st.markdown("### 🔮 LSTM 净值预测走势")

    hist_part = df.tail(60)
    pred_dates = pd.to_datetime(result["lstm_dates"])
    pred_vals = result["lstm_pred"]

    fig_pred = go.Figure()

    # 历史净值线
    fig_pred.add_trace(
        go.Scatter(
            x=hist_part["date"], y=hist_part["nav"],
            mode="lines", name="📜 历史净值",
            line=dict(color="#1f77b4", width=2),
        )
    )

    # MA20 参考线
    fig_pred.add_trace(
        go.Scatter(
            x=hist_part["date"], y=hist_part["MA20"],
            mode="lines", name="📊 MA20 均线",
            line=dict(color="#2ca02c", width=1, dash="dot"),
        )
    )

    # 预测线
    last_hist_val = hist_part["nav"].iloc[-1]
    all_pred_vals = [last_hist_val] + list(pred_vals)
    all_pred_dates = [hist_part["date"].iloc[-1]] + list(pred_dates)

    fig_pred.add_trace(
        go.Scatter(
            x=all_pred_dates, y=all_pred_vals,
            mode="lines+markers+text",
            name="🤖 LSTM 预测",
            line=dict(color="#ff7f0e", width=2.5, dash="dash"),
            marker=dict(size=10, symbol="circle", color="#ff7f0e"),
            text=[f"{v:.4f}" for v in all_pred_vals],
            textposition="top center",
            textfont=dict(size=10, color="#ff7f0e"),
        )
    )

    # 置信区间
    std_est = hist_part["change_pct"].std() / 100 * np.array(all_pred_vals)
    upper = np.array(all_pred_vals) + std_est * 2
    lower = np.array(all_pred_vals) - std_est * 2
    fig_pred.add_trace(
        go.Scatter(
            x=list(all_pred_dates) + list(all_pred_dates)[::-1],
            y=list(upper) + list(lower)[::-1],
            fill="toself", fillcolor="rgba(255,127,14,0.12)",
            line=dict(color="rgba(255,127,14,0)"),
            name="📏 置信区间 (±2σ)",
        )
    )

    # 分界线
    split_date = hist_part["date"].iloc[-1]
    y_mid = (hist_part["nav"].max() + hist_part["nav"].min()) / 2
    fig_pred.add_vline(x=split_date, line_dash="dot", line_color="gray", line_width=1.5)
    fig_pred.add_annotation(
        x=split_date, y=hist_part["nav"].max(),
        text="← 历史 | 预测 →", showarrow=False,
        font=dict(color="gray", size=11),
    )

    # 布林带叠加到预测图
    fig_pred.add_trace(
        go.Scatter(
            x=hist_part["date"], y=hist_part["BB_upper"],
            mode="lines", name="布林上轨",
            line=dict(color="rgba(239,83,80,0.4)", width=0.8, dash="dot"),
            showlegend=True,
        )
    )
    fig_pred.add_trace(
        go.Scatter(
            x=hist_part["date"], y=hist_part["BB_lower"],
            mode="lines", name="布林下轨",
            line=dict(color="rgba(38,166,154,0.4)", width=0.8, dash="dot"),
            showlegend=True,
        )
    )

    fig_pred.update_layout(
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=10),
    )
    st.plotly_chart(fig_pred, width="stretch")

    # ---- 逐日预测明细 ----
    st.markdown("### 📋 逐日预测明细")
    pred_table = pd.DataFrame({
        "日期": result["lstm_dates"],
        "预测净值": result["lstm_pred"],
        "较前日涨跌": [
            round(result["lstm_pred"][0] - result["last_nav"], 4)
        ] + [
            round(result["lstm_pred"][i] - result["lstm_pred"][i - 1], 4)
            for i in range(1, len(result["lstm_pred"]))
        ],
        "累计变化%": [
            round((v - result["last_nav"]) / result["last_nav"] * 100, 2)
            for v in result["lstm_pred"]
        ],
    })
    # 涨跌上色
    def color_change(val):
        if isinstance(val, (int, float)):
            return "color: #26a69a" if val > 0 else ("color: #ef5350" if val < 0 else "")
        return ""
    st.dataframe(
        pred_table.style.map(color_change, subset=["较前日涨跌", "累计变化%"]),
        width="stretch",
    )

    # ---- 涨跌概率仪表盘 ----
    st.markdown("### 🎛️ 涨跌概率分布")
    prob_col1, prob_col2, prob_col3 = st.columns([1, 2, 1])
    with prob_col2:
        up_prob = result["xgb_up_prob"] * 100
        down_prob = 100 - up_prob
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=up_prob,
            title={"text": "XGBoost 次日上涨概率 (%)"},
            delta={"reference": 50, "increasing": {"color": "#26a69a"}, "decreasing": {"color": "#ef5350"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#26a69a" if up_prob > 50 else "#ef5350"},
                "steps": [
                    {"range": [0, 30], "color": "rgba(239,83,80,0.3)"},
                    {"range": [30, 50], "color": "rgba(255,193,7,0.3)"},
                    {"range": [50, 70], "color": "rgba(255,193,7,0.3)"},
                    {"range": [70, 100], "color": "rgba(38,166,154,0.3)"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 2},
                    "thickness": 0.8, "value": 50,
                },
            },
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, width="stretch")

    # ---- 综合研判 ----
    st.markdown("### 📝 综合研判报告")

    lstm_trend = result["lstm_trend"]
    xgb_prob = result["xgb_up_prob"]
    confidence = result["confidence_note"]

    if lstm_trend == "看涨" and xgb_prob > 0.5:
        overall = "✅ 偏多"
        detail = f"LSTM 预测{lstm_trend}，XGBoost 上涨概率 {xgb_prob*100:.1f}%，两者方向一致。"
    elif lstm_trend == "看跌" and xgb_prob < 0.5:
        overall = "❌ 偏空"
        detail = f"LSTM 预测{lstm_trend}，XGBoost 上涨概率仅 {xgb_prob*100:.1f}%，两者方向一致。"
    else:
        overall = "⚠️ 分歧"
        detail = f"LSTM 趋势为「{lstm_trend}」，但 XGBoost 次日上涨概率为 {xgb_prob*100:.1f}%，模型间存在分歧，建议观望。"

    # 技术面综合
    last_rsi = df["RSI"].iloc[-1]
    macd_hist = df["MACD_hist"].iloc[-1]
    tech_signal = ""
    if not pd.isna(last_rsi):
        if last_rsi > 70:
            tech_signal += f"RSI={last_rsi:.1f}（超买区）⚠️；"
        elif last_rsi < 30:
            tech_signal += f"RSI={last_rsi:.1f}（超卖区）💡；"
        else:
            tech_signal += f"RSI={last_rsi:.1f}（中性）；"
    if not pd.isna(macd_hist):
        tech_signal += f"MACD柱={'正值' if macd_hist > 0 else '负值'}（{'多头' if macd_hist > 0 else '空头'}信号）"

    st.info(f"""
**综合研判：{overall}**（置信度：{confidence}）

**技术面**：{tech_signal}

**模型预测**：{detail}

**关键位参考**：
- 上方阻力：{resist:.4f}（布林上轨）
- 下方支撑：{support:.4f}（布林下轨）
- 当前距阻力 {((resist - result['last_nav']) / result['last_nav'] * 100):.2f}%，距支撑 {((result['last_nav'] - support) / result['last_nav'] * 100):.2f}%

⚠️ 再次提醒：以上分析完全基于历史数据的统计规律和机器学习拟合，**不能预测政策变化、市场情绪、突发事件**等关键因素。请仅作为学习参考，实际投资请咨询专业顾问。
    """)

    # ---- 导出离线报告 ----
    st.markdown("---")
    st.markdown("### 📥 离线报告")

    if st.button("📄 生成离线 HTML 报告", type="secondary"):
        with st.spinner("正在生成报告..."):
            report_path = generate_offline_report(
                fund_name=fund_name, code=code,
                df=df, df_signal=df_signal,
                result=result,
                hist_vol=hist_vol, max_drawdown=max_drawdown,
                recent_high=recent_high, recent_low=recent_low,
                support=support, resist=resist,
                overall=overall, detail=detail,
                tech_signal=tech_signal,
                confidence=confidence,
            )
            st.success(f"✅ 报告已生成：`{report_path}`")

            # 提供下载
            with open(report_path, "rb") as f:
                st.download_button(
                    "⬇️ 下载报告到手机",
                    data=f,
                    file_name=f"fund_report_{code}.html",
                    mime="text/html",
                )

            st.info("💡 **离线使用**：下载 .html 文件到手机后，用浏览器打开即可查看，无需联网、无需开电脑。")

# ============================================================
# 页脚
# ============================================================

st.markdown("---")
st.caption(
    "📈 基金涨幅预测系统 | 数据来源: akshare | "
    "技术指标: pandas-ta | ML: PyTorch LSTM + XGBoost | "
    "⚠️ 仅供学习参考，不构成投资建议"
)
