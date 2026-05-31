"""
财经新闻抓取模块 — 从东方财富 / 新浪财经获取基金相关新闻和市场事件。
"""

import json
import os
import time
from datetime import datetime, timedelta

import requests

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
NEWS_CACHE_FILE = os.path.join(CACHE_DIR, "news_cache.json")


def _load_news_cache() -> dict:
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(NEWS_CACHE_FILE):
        try:
            with open(NEWS_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 缓存 2 小时有效
                if time.time() - data.get("timestamp", 0) < 7200:
                    return data.get("news", {})
        except Exception:
            pass
    return {}


def _save_news_cache(news: dict):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(NEWS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump({"timestamp": time.time(), "news": news}, f, ensure_ascii=False)


def fetch_fund_news(code: str, fund_name: str = "", limit: int = 15) -> list[dict]:
    """
    抓取基金相关新闻。

    参数:
        code: 基金代码
        fund_name: 基金名称（可选，用于过滤）
        limit: 返回新闻条数

    返回:
        list[dict]: title, time, source, url, summary
    """
    cache = _load_news_cache()
    cache_key = f"fund_{code}"
    if cache_key in cache:
        return cache[cache_key][:limit]

    news_list = []

    try:
        # 东方财富 — 基金新闻搜索
        url = "https://search-api-web.eastmoney.com/search/jsonp"
        params = {
            "cb": "",
            "param": json.dumps({
                "uid": "",
                "keyword": fund_name or code,
                "type": ["8193"],  # 新闻
                "client": "web",
                "pageIndex": 1,
                "pageSize": limit,
            }),
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://so.eastmoney.com/",
        }
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code == 200:
            # 解析 JSONP
            text = resp.text.strip()
            if text.startswith("("):
                text = text[1:-1]
            data = json.loads(text)
            items = data.get("Data", {}).get("Data", [])
            for item in items:
                news_list.append({
                    "title": item.get("Title", ""),
                    "time": item.get("ShowDate", ""),
                    "source": item.get("InfoCode", "东方财富"),
                    "url": item.get("Url", ""),
                    "summary": item.get("Content", "").replace("<em>", "").replace("</em>", "")[:200],
                })
    except Exception:
        pass

    # 如果上面没抓到，尝试新浪财经
    if not news_list:
        try:
            keyword = fund_name or code
            url = f"https://feed.mix.sina.com.cn/api/roll/get"
            params = {
                "pageid": 153,
                "lid": 2512,
                "k": keyword,
                "num": limit,
                "page": 1,
            }
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("result", {}).get("data", [])
                for item in items:
                    news_list.append({
                        "title": item.get("title", ""),
                        "time": item.get("ctime", ""),
                        "source": item.get("media_name", "新浪财经"),
                        "url": item.get("url", ""),
                        "summary": item.get("intro", "")[:200],
                    })
        except Exception:
            pass

    # 如果还是没有，用 akshare 新闻接口
    if not news_list:
        try:
            import akshare as ak
            df = ak.stock_news_em()  # 全市场新闻
            if df is not None and not df.empty:
                keyword = (fund_name or code).lower()
                for _, row in df.head(limit * 3).iterrows():
                    title = str(row.get("标题", row.get("title", "")))
                    if keyword.lower() in title.lower() or any(
                        kw in title for kw in ["基金", "ETF", "指数"]
                    ):
                        news_list.append({
                            "title": title,
                            "time": str(row.get("发布时间", "")),
                            "source": "东方财富",
                            "url": str(row.get("链接", row.get("url", ""))),
                            "summary": str(row.get("内容", ""))[:200],
                        })
                news_list = news_list[:limit]
        except Exception:
            pass

    # 如果最终还是没有新闻，返回一些通用市场消息
    if not news_list:
        news_list = [{
            "title": "暂无该基金相关新闻",
            "time": datetime.now().strftime("%Y-%m-%d"),
            "source": "系统",
            "url": "",
            "summary": "当前时段未抓取到相关资讯，请稍后重试或检查基金代码",
        }]

    # 缓存
    cache[cache_key] = news_list
    _save_news_cache(cache)

    return news_list[:limit]


def fetch_market_overview() -> list[dict]:
    """获取市场概况新闻（宏观层面）。"""
    cache = _load_news_cache()
    if "market" in cache:
        return cache["market"]

    news_list = []
    try:
        import akshare as ak
        df = ak.stock_news_em()
        if df is not None and not df.empty:
            for _, row in df.head(20).iterrows():
                title = str(row.get("标题", row.get("title", "")))
                if any(kw in title for kw in ["A股", "沪指", "央行", "政策", "市场", "降息", "利率", "经济"]):
                    news_list.append({
                        "title": title,
                        "time": str(row.get("发布时间", "")),
                        "source": "东方财富",
                        "url": str(row.get("链接", "")),
                        "summary": "",
                    })
    except Exception:
        pass

    if not news_list:
        news_list = [{"title": "暂无市场要闻", "time": "", "source": "", "url": "", "summary": ""}]

    cache["market"] = news_list
    _save_news_cache(cache)
    return news_list[:10]
