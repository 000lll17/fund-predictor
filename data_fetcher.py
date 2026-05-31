"""
数据获取模块 — 基于 akshare 获取基金净值/行情数据。
支持公募基金和 ETF，自动识别类型并缓存数据。
"""

import os
from datetime import datetime, timedelta

import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(code: str) -> str:
    return os.path.join(CACHE_DIR, f"{code}.csv")


def _load_cache(code: str, max_age_days: int = 1) -> pd.DataFrame | None:
    """加载本地缓存，缓存有效期默认 1 天。"""
    path = _cache_path(code)
    if not os.path.exists(path):
        return None
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    if datetime.now() - mtime > timedelta(days=max_age_days):
        return None
    try:
        df = pd.read_csv(path, parse_dates=["date"], index_col=0)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def _save_cache(code: str, df: pd.DataFrame):
    _ensure_cache_dir()
    df.to_csv(_cache_path(code))


def _is_etf(code: str) -> bool:
    """简单判断：5位数字且不以00/16开头 → 大概率是场内 ETF/LOF。"""
    code = code.strip()
    if not code.isdigit():
        return False
    if len(code) <= 5:
        # 5位及以下 → 场内代码（ETF/LOF）
        return True
    # 6位数字 → 公募基金代码
    return False


def _fetch_mutual_fund(code: str) -> pd.DataFrame:
    """获取公募基金单位净值走势。"""
    import akshare as ak

    try:
        df = ak.fund_open_fund_info_em(
            symbol=code, indicator="单位净值走势"
        )
    except Exception:
        # 尝试另一个接口
        df = ak.fund_open_fund_info_em(
            symbol=code, indicator="累计净值走势"
        )

    if df is None or df.empty:
        raise ValueError(f"未找到基金 {code} 的数据，请检查代码是否正确")

    # 标准化列名
    df.columns = [c.strip() for c in df.columns]
    # akshare 通常返回：净值日期, 单位净值, 日增长率
    col_map = {}
    for c in df.columns:
        if "日期" in c:
            col_map[c] = "date"
        elif "单位净值" in c:
            col_map[c] = "nav"
        elif "累计净值" in c:
            col_map[c] = "acc_nav"
        elif "增长" in c or "涨" in c:
            col_map[c] = "change_pct"

    df = df.rename(columns=col_map)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 确保必要列存在
    if "nav" not in df.columns:
        raise ValueError(f"基金 {code} 数据缺少单位净值列，可用列: {list(df.columns)}")
    if "change_pct" not in df.columns:
        # 手动计算
        df["change_pct"] = df["nav"].pct_change() * 100
    df["volume"] = 0  # 公募基金无成交量
    df["change_pct"] = df["change_pct"].fillna(0)

    return df[["date", "nav", "change_pct", "volume"]]


def _fetch_etf(code: str) -> pd.DataFrame:
    """获取 ETF 历史行情数据。"""
    import akshare as ak

    df = ak.fund_etf_hist_em(
        symbol=code,
        period="daily",
        start_date="20100101",
        end_date=datetime.now().strftime("%Y%m%d"),
        adjust="qfq",  # 前复权
    )

    if df is None or df.empty:
        raise ValueError(f"未找到 ETF {code} 的数据，请检查代码是否正确")

    # 标准化列名
    col_map = {
        "日期": "date",
        "收盘": "close",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "涨跌幅": "change_pct",
    }
    df = df.rename(columns=col_map)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ETF 用收盘价作为"净值"
    df["nav"] = df["close"]
    if "change_pct" not in df.columns:
        df["change_pct"] = df["nav"].pct_change() * 100
    if "volume" not in df.columns:
        df["volume"] = 0

    df["change_pct"] = df["change_pct"].fillna(0)
    df["volume"] = df["volume"].fillna(0)

    return df[["date", "nav", "change_pct", "volume"]]


def fetch_fund_nav(
    code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    获取基金净值数据。

    参数:
        code: 基金代码，如 000001（华夏成长混合）、510050（上证50ETF）
        start_date: 起始日期 'YYYYMMDD' 或 'YYYY-MM-DD'，默认 5 年前
        end_date: 截止日期，默认今天
        force_refresh: 是否强制刷新缓存

    返回:
        DataFrame，列: date, nav, change_pct, volume
    """
    code = str(code).strip().zfill(6)

    # 尝试缓存
    if not force_refresh:
        cached = _load_cache(code)
        if cached is not None:
            df = cached
        else:
            if _is_etf(code):
                df = _fetch_etf(code)
            else:
                df = _fetch_mutual_fund(code)
            _save_cache(code, df)
    else:
        if _is_etf(code):
            df = _fetch_etf(code)
        else:
            df = _fetch_mutual_fund(code)
        _save_cache(code, df)

    # 日期过滤
    if start_date:
        start = pd.Timestamp(start_date.replace("-", "")[:8])
        df = df[df["date"] >= start]
    if end_date:
        end = pd.Timestamp(end_date.replace("-", "")[:8])
        df = df[df["date"] <= end]

    return df.reset_index(drop=True)


def get_fund_name(code: str) -> str:
    """获取基金名称。"""
    import akshare as ak

    code = str(code).strip().zfill(6)

    try:
        if _is_etf(code):
            df = ak.fund_etf_fund_info_em()
            # ETF 信息表列名可能不同，尝试匹配
            for col in ["基金代码", "code", "证券代码"]:
                if col in df.columns:
                    match = df[df[col].astype(str).str.strip() == code]
                    if not match.empty:
                        for name_col in ["基金简称", "name", "证券简称"]:
                            if name_col in match.columns:
                                return str(match.iloc[0][name_col])
                    break
        else:
            df = ak.fund_open_fund_info_em(symbol=code, indicator="基金信息")
            if df is not None and not df.empty:
                # 基金名称通常在第一个单元格
                return str(df.iloc[0, 0])
    except Exception:
        pass

    return f"基金{code}"


def clear_cache(code: str | None = None):
    """清除缓存。若 code 为 None 则清除全部。"""
    _ensure_cache_dir()
    if code is None:
        for f in os.listdir(CACHE_DIR):
            os.remove(os.path.join(CACHE_DIR, f))
    else:
        path = _cache_path(str(code).strip().zfill(6))
        if os.path.exists(path):
            os.remove(path)
