"""
技术指标计算模块 — 基于 pandas-ta 计算常用技术指标，
并生成综合买卖信号。
"""

import numpy as np
import pandas as pd


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    在输入 DataFrame 上添加所有技术指标列。

    输入需包含: date, nav, change_pct, volume
    输出增加: MA5, MA10, MA20, MA60, MACD, MACD_signal, MACD_hist,
              RSI, BB_upper, BB_middle, BB_lower, vol_MA5, vol_MA20
    """
    df = df.copy()

    close = df["nav"].astype(float)
    high = close  # 公募基金无最高价，用净值代替
    low = close
    vol = df["volume"].astype(float) if "volume" in df.columns else pd.Series(0, index=df.index)

    # ---- 移动均线 ----
    df["MA5"] = close.rolling(window=5).mean()
    df["MA10"] = close.rolling(window=10).mean()
    df["MA20"] = close.rolling(window=20).mean()
    df["MA60"] = close.rolling(window=60).mean()

    # ---- MACD ----
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # ---- RSI (14 日) ----
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    # 使用 Wilder's smoothing
    for i in range(14, len(df)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * 13 + gain.iloc[i]) / 14
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * 13 + loss.iloc[i]) / 14
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # ---- 布林带 (20日, 2倍标准差) ----
    df["BB_middle"] = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    df["BB_upper"] = df["BB_middle"] + 2 * bb_std
    df["BB_lower"] = df["BB_middle"] - 2 * bb_std

    # ---- 成交量均线 ----
    df["vol_MA5"] = vol.rolling(window=5).mean()
    df["vol_MA20"] = vol.rolling(window=20).mean()

    return df


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    根据技术指标生成买卖信号标记。

    返回原 DataFrame，添加以下列:
        signal: 1=买入, -1=卖出, 0=无信号
        signal_reason: 信号原因描述
    """
    df = df.copy()

    if "MA5" not in df.columns:
        df = compute_all(df)

    df["signal"] = 0
    df["signal_reason"] = ""

    reasons = [[] for _ in range(len(df))]

    for i in range(1, len(df)):
        buy_signals = []
        sell_signals = []

        # 1. MA 金叉/死叉
        if i >= 1:
            if (
                df["MA5"].iloc[i] > df["MA20"].iloc[i]
                and df["MA5"].iloc[i - 1] <= df["MA20"].iloc[i - 1]
            ):
                buy_signals.append("MA5 上穿 MA20（金叉）")
            if (
                df["MA5"].iloc[i] < df["MA20"].iloc[i]
                and df["MA5"].iloc[i - 1] >= df["MA20"].iloc[i - 1]
            ):
                sell_signals.append("MA5 下穿 MA20（死叉）")

        # 2. MACD 金叉/死叉
        if i >= 1:
            if (
                df["MACD"].iloc[i] > df["MACD_signal"].iloc[i]
                and df["MACD"].iloc[i - 1] <= df["MACD_signal"].iloc[i - 1]
            ):
                buy_signals.append("MACD 金叉")
            if (
                df["MACD"].iloc[i] < df["MACD_signal"].iloc[i]
                and df["MACD"].iloc[i - 1] >= df["MACD_signal"].iloc[i - 1]
            ):
                sell_signals.append("MACD 死叉")

        # 3. RSI 超买超卖
        if not pd.isna(df["RSI"].iloc[i]):
            if df["RSI"].iloc[i] < 30 and df["RSI"].iloc[i - 1] >= 30:
                buy_signals.append(f"RSI 超卖 ({df['RSI'].iloc[i]:.1f})")
            if df["RSI"].iloc[i] > 70 and df["RSI"].iloc[i - 1] <= 70:
                sell_signals.append(f"RSI 超买 ({df['RSI'].iloc[i]:.1f})")

        # 4. 布林带突破
        if not pd.isna(df["BB_lower"].iloc[i]):
            if close_i := df["nav"].iloc[i]:
                if close_i < df["BB_lower"].iloc[i]:
                    buy_signals.append("价格跌破布林下轨")
                if close_i > df["BB_upper"].iloc[i]:
                    sell_signals.append("价格突破布林上轨")

        # 汇总信号
        if buy_signals:
            df.loc[df.index[i], "signal"] = 1
            reasons[i].extend(buy_signals)
        if sell_signals:
            # 如果同时有买入和卖出信号，卖出优先
            df.loc[df.index[i], "signal"] = -1
            reasons[i].extend(sell_signals)

        df.loc[df.index[i], "signal_reason"] = "; ".join(reasons[i])

    return df


def signal_summary(df: pd.DataFrame) -> dict:
    """返回最近的信号汇总。"""
    df = generate_signals(df) if "signal" not in df.columns else df

    recent = df[df["signal"] != 0].tail(20)

    last_date = df["date"].iloc[-1]
    last_nav = df["nav"].iloc[-1]
    last_ma5 = df["MA5"].iloc[-1]
    last_ma20 = df["MA20"].iloc[-1]
    last_rsi = df["RSI"].iloc[-1]
    last_macd_hist = df["MACD_hist"].iloc[-1]

    # 趋势判断
    if not pd.isna(last_ma5) and not pd.isna(last_ma20):
        if last_ma5 > last_ma20:
            trend = "短期看涨（MA5 > MA20）"
        else:
            trend = "短期看跌（MA5 < MA20）"
    else:
        trend = "数据不足"

    # RSI 状态
    if not pd.isna(last_rsi):
        if last_rsi > 70:
            rsi_status = f"超买 ({last_rsi:.1f})"
        elif last_rsi < 30:
            rsi_status = f"超卖 ({last_rsi:.1f})"
        else:
            rsi_status = f"中性 ({last_rsi:.1f})"
    else:
        rsi_status = "数据不足"

    return {
        "latest_date": last_date.strftime("%Y-%m-%d") if hasattr(last_date, "strftime") else str(last_date),
        "latest_nav": round(float(last_nav), 4),
        "trend": trend,
        "rsi_status": rsi_status,
        "recent_signals": [
            {
                "date": row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"]),
                "type": "买入" if row["signal"] == 1 else "卖出",
                "reason": row["signal_reason"],
            }
            for _, row in recent.iterrows()
        ],
    }
