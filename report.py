"""
离线 HTML 报告生成器 — 生成包含所有图表和分析的自包含 HTML 文件。
"""

import base64
import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _fig_to_base64(fig: go.Figure) -> str:
    """将 Plotly 图表转为 Base64 字符串，可嵌入 HTML。"""
    buf = io.BytesIO()
    fig.write_image(buf, format="png", width=1200, height=500, scale=1.5)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _fig_to_html_div(fig: go.Figure, div_id: str) -> str:
    """将图表转成内嵌 plotly.js 的 HTML div。"""
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


def _metrics_card(title: str, value: str, delta: str = "", color: str = "") -> str:
    """生成指标卡片 HTML。"""
    delta_html = f'<span class="delta" style="color:{color}">{delta}</span>' if delta else ""
    return f"""
    <div class="card">
        <div class="card-title">{title}</div>
        <div class="card-value">{value}</div>
        {delta_html}
    </div>"""


def generate_offline_report(
    fund_name: str,
    code: str,
    df: pd.DataFrame,
    df_signal: pd.DataFrame,
    result: dict,
    hist_vol: float,
    max_drawdown: float,
    recent_high: float,
    recent_low: float,
    support: float,
    resist: float,
    overall: str,
    detail: str,
    tech_signal: str,
    confidence: str,
) -> str:
    """生成完整的离线 HTML 报告，返回文件路径。"""

    # ---- 构建图表 ----

    # Chart 1: 净值走势 + 预测
    hist_part = df.tail(60)
    pred_dates = pd.to_datetime(result["lstm_dates"])
    pred_vals = result["lstm_pred"]

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=hist_part["date"], y=hist_part["nav"],
        mode="lines", name="历史净值", line=dict(color="#1f77b4", width=2),
    ))
    fig1.add_trace(go.Scatter(
        x=hist_part["date"], y=hist_part["MA20"],
        mode="lines", name="MA20", line=dict(color="#2ca02c", width=1, dash="dot"),
    ))

    last_hist_val = hist_part["nav"].iloc[-1]
    all_pred_vals = [last_hist_val] + list(pred_vals)
    all_pred_dates = [hist_part["date"].iloc[-1]] + list(pred_dates)

    fig1.add_trace(go.Scatter(
        x=all_pred_dates, y=all_pred_vals,
        mode="lines+markers", name="LSTM 预测",
        line=dict(color="#ff7f0e", width=2.5, dash="dash"),
        marker=dict(size=8),
    ))

    # 置信区间
    std_est = hist_part["change_pct"].std() / 100 * np.array(all_pred_vals)
    upper = np.array(all_pred_vals) + std_est * 2
    lower = np.array(all_pred_vals) - std_est * 2
    fig1.add_trace(go.Scatter(
        x=list(all_pred_dates) + list(all_pred_dates)[::-1],
        y=list(upper) + list(lower)[::-1],
        fill="toself", fillcolor="rgba(255,127,14,0.12)",
        line=dict(color="rgba(255,127,14,0)"),
        name="置信区间",
    ))

    fig1.update_layout(
        title=f"{fund_name}（{code}）净值走势 & 预测",
        height=500, hovermode="x unified",
        template="plotly_white",
    )

    # Chart 2: MACD
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.5])
    fig2.add_trace(go.Scatter(
        x=df["date"], y=df["nav"], mode="lines", name="净值",
        line=dict(color="#1f77b4"),
    ), row=1, col=1)
    fig2.add_trace(go.Scatter(
        x=df["date"], y=df["MACD"], mode="lines", name="MACD",
        line=dict(color="#1f77b4"),
    ), row=2, col=1)
    fig2.add_trace(go.Scatter(
        x=df["date"], y=df["MACD_signal"], mode="lines", name="Signal",
        line=dict(color="#ff7f0e"),
    ), row=2, col=1)
    colors_hist = ["#ef5350" if v < 0 else "#26a69a" for v in df["MACD_hist"].fillna(0)]
    fig2.add_trace(go.Bar(
        x=df["date"], y=df["MACD_hist"], name="Histogram",
        marker_color=colors_hist, opacity=0.5,
    ), row=2, col=1)
    fig2.update_layout(height=400, title="MACD 趋势指标", template="plotly_white")

    # Chart 3: RSI
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df["date"], y=df["RSI"], mode="lines", name="RSI(14)",
        line=dict(color="#9467bd", width=2),
    ))
    fig3.add_hline(y=70, line_dash="dash", line_color="red")
    fig3.add_hline(y=30, line_dash="dash", line_color="green")
    fig3.update_layout(height=300, title="RSI 相对强弱", template="plotly_white", yaxis_range=[0, 100])

    # 将所有图表转为 HTML + plotly.js
    plotly_js_cdn = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
    charts_html = (
        plotly_js_cdn
        + fig1.to_html(full_html=False, include_plotlyjs=False)
        + fig2.to_html(full_html=False, include_plotlyjs=False)
        + fig3.to_html(full_html=False, include_plotlyjs=False)
    )

    # ---- 预测明细表 ----
    pred_rows = ""
    for i in range(len(result["lstm_dates"])):
        change = round(result["lstm_pred"][i] - result["last_nav"], 4) if i == 0 else round(result["lstm_pred"][i] - result["lstm_pred"][i-1], 4)
        cumul = round((result["lstm_pred"][i] - result["last_nav"]) / result["last_nav"] * 100, 2)
        color = "#26a69a" if cumul > 0 else ("#ef5350" if cumul < 0 else "#666")
        pred_rows += f"""
        <tr>
            <td>{result['lstm_dates'][i]}</td>
            <td>{result['lstm_pred'][i]:.4f}</td>
            <td style="color:{color}">{change:+.4f}</td>
            <td style="color:{color}">{cumul:+.2f}%</td>
        </tr>"""

    # ---- 组装完整 HTML ----
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{fund_name}（{code}）基金分析报告</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
            background: #f5f5f5; color: #333; line-height: 1.6;
        }}
        .header {{
            background: linear-gradient(135deg, #1f77b4 0%, #2ca02c 100%);
            color: white; padding: 30px 20px; text-align: center;
        }}
        .header h1 {{ font-size: 24px; margin-bottom: 5px; }}
        .header .code {{ font-size: 14px; opacity: 0.85; }}
        .header .date {{ font-size: 12px; opacity: 0.7; margin-top: 5px; }}
        .warning {{
            background: #fff3cd; border: 1px solid #ffc107; color: #856404;
            padding: 12px 16px; margin: 16px; border-radius: 8px;
            font-size: 13px; text-align: center;
        }}
        .container {{ max-width: 800px; margin: 0 auto; padding: 0 16px 30px; }}
        .cards {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin: 16px 0; }}
        .card {{
            background: white; border-radius: 10px; padding: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .card-title {{ font-size: 12px; color: #888; margin-bottom: 6px; }}
        .card-value {{ font-size: 22px; font-weight: bold; color: #1f77b4; }}
        .delta {{ font-size: 13px; display: block; margin-top: 2px; }}
        .chart-container {{
            background: white; border-radius: 10px; padding: 12px;
            margin: 16px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .section-title {{
            font-size: 18px; font-weight: bold; color: #333;
            margin: 20px 0 12px; padding-left: 12px;
            border-left: 4px solid #1f77b4;
        }}
        table {{
            width: 100%; border-collapse: collapse; margin: 12px 0;
            background: white; border-radius: 10px; overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        th {{ background: #1f77b4; color: white; padding: 10px; font-size: 13px; text-align: center; }}
        td {{ padding: 10px; text-align: center; border-bottom: 1px solid #eee; font-size: 13px; }}
        .verdict {{
            background: white; border-radius: 10px; padding: 20px;
            margin: 16px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border-left: 4px solid #1f77b4;
        }}
        .verdict h3 {{ color: #1f77b4; margin-bottom: 10px; }}
        .footer {{
            text-align: center; color: #999; font-size: 12px;
            padding: 20px; border-top: 1px solid #eee;
        }}
    </style>
</head>
<body>

<div class="header">
    <h1>{fund_name}</h1>
    <div class="code">基金代码：{code}</div>
    <div class="date">报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
</div>

<div class="warning">
    ⚠️ <strong>重要声明</strong>：本报告所有分析和预测结果仅供学习参考，<strong>不构成任何投资建议</strong>。
    基金投资存在亏损风险，请根据自身风险承受能力独立决策。过往业绩不代表未来表现。
</div>

<div class="container">

    <div class="section-title">📊 预测概览</div>
    <div class="cards">
        {_metrics_card("当前净值", f"{result['last_nav']:.4f}")}
        {_metrics_card("预测趋势", result['lstm_trend'], "", "#26a69a" if result['lstm_trend'] == '看涨' else "#ef5350")}
        {_metrics_card("次日涨概率", f"{result['xgb_up_prob']*100:.1f}%", "", "#26a69a" if result['xgb_up_prob'] > 0.5 else "#ef5350")}
        {_metrics_card("预测变化", f"{result['pred_change_pct']:+.2f}%", f"{len(result['lstm_pred'])}日累计", "#26a69a" if result['pred_change_pct'] > 0 else "#ef5350")}
    </div>

    <div class="section-title">📉 风险指标</div>
    <div class="cards">
        {_metrics_card("历史波动率", f"{hist_vol:.2f}%")}
        {_metrics_card("最大回撤", f"{max_drawdown:.2f}%")}
        {_metrics_card("60日高点", f"{recent_high:.4f}")}
        {_metrics_card("60日低点", f"{recent_low:.4f}")}
    </div>

    <div class="section-title">🎯 关键价位</div>
    <div class="cards">
        {_metrics_card("阻力位", f"{resist:.4f}", f"距当前 {((resist - result['last_nav']) / result['last_nav'] * 100):.2f}%")}
        {_metrics_card("支撑位", f"{support:.4f}", f"距当前 {((result['last_nav'] - support) / result['last_nav'] * 100):.2f}%")}
        {_metrics_card("置信度", confidence)}
        {_metrics_card("RSI", f"{df['RSI'].iloc[-1]:.1f}" if not pd.isna(df['RSI'].iloc[-1]) else "N/A")}
    </div>

    <div class="section-title">🔮 净值走势 & 预测</div>
    <div class="chart-container">{fig1.to_html(full_html=False, include_plotlyjs=False)}</div>

    <div class="section-title">📉 MACD 趋势指标</div>
    <div class="chart-container">{fig2.to_html(full_html=False, include_plotlyjs=False)}</div>

    <div class="section-title">📊 RSI 相对强弱</div>
    <div class="chart-container">{fig3.to_html(full_html=False, include_plotlyjs=False)}</div>

    <div class="section-title">📋 逐日预测</div>
    <table>
        <tr><th>日期</th><th>预测净值</th><th>涨跌</th><th>累计</th></tr>
        {pred_rows}
    </table>

    <div class="section-title">📝 综合研判</div>
    <div class="verdict">
        <h3>{overall}</h3>
        <p><strong>技术面</strong>：{tech_signal}</p>
        <p><strong>模型预测</strong>：{detail}</p>
        <p><strong>置信度</strong>：{confidence}</p>
        <p style="margin-top:12px;color:#856404;font-size:13px;">
            ⚠️ 以上分析基于历史数据统计规律和机器学习拟合，不能预测政策变化、市场情绪、突发事件等因素。
        </p>
    </div>

</div>

<div class="footer">
    📈 基金涨幅预测系统 | 生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 仅供学习参考
</div>

{plotly_js_cdn}
</body>
</html>"""

    # 保存文件
    report_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"report_{code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    return report_path
