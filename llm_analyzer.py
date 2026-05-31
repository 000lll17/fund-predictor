"""
LLM 智能分析模块 — 基于 Ollama + Qwen 2.5 进行新闻情绪分析、事件解读、AI 报告生成。
"""

import json
import os
import time
from datetime import datetime
from typing import Optional

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
LLM_CACHE_FILE = os.path.join(CACHE_DIR, "llm_cache.json")


def _load_llm_cache() -> dict:
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(LLM_CACHE_FILE):
        try:
            with open(LLM_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 缓存 4 小时
                if time.time() - data.get("timestamp", 0) < 14400:
                    return data.get("results", {})
        except Exception:
            pass
    return {}


def _save_llm_cache(results: dict):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(LLM_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump({"timestamp": time.time(), "results": results}, f, ensure_ascii=False)


def _ollama_generate(prompt: str, system: str = "", timeout: int = 120) -> str:
    """调用 Ollama API 生成文本。"""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 1024,
        },
    }
    if system:
        payload["system"] = system

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        if resp.status_code == 200:
            return resp.json().get("response", "")
        else:
            return f"[错误] Ollama 返回状态码 {resp.status_code}"
    except requests.exceptions.ConnectionError:
        return "[错误] 无法连接 Ollama，请确保已安装并启动（运行 setup.bat）"
    except Exception as e:
        return f"[错误] {e}"


def check_ollama_available() -> tuple[bool, str]:
    """检查 Ollama 服务是否可用。"""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            return True, f"已就绪（可用模型: {', '.join(model_names[:3])}）"
        return False, "Ollama 未响应"
    except Exception:
        return False, "Ollama 未启动，请运行 'ollama serve' 或双击 setup.bat"


def analyze_news_sentiment(news_list: list[dict]) -> dict:
    """分析新闻情绪和关键事件。返回情绪得分和事件摘要。"""
    if not news_list or (len(news_list) == 1 and "暂无" in news_list[0].get("title", "")):
        return {
            "sentiment_score": 0,
            "sentiment_label": "无新闻数据",
            "key_events": [],
            "summary": "当前无相关新闻可供分析",
        }

    # 缓存检查
    cache = _load_llm_cache()
    news_key = "sent_" + str(hash(json.dumps(news_list, ensure_ascii=False)))
    if news_key in cache:
        return cache[news_key]

    titles = "\n".join([
        f"{i+1}. [{n.get('source','')}] {n.get('title','')}（{n.get('time','')}）"
        for i, n in enumerate(news_list[:15])
    ])

    prompt = f"""请分析以下基金相关新闻，输出 JSON 格式结果：

新闻列表：
{titles}

请以 JSON 格式返回（直接返回 JSON，不要其他文字）：
{{
    "sentiment_score": 0.0（-1到1之间的浮点数，-1最负面 +1最正面 0中性),
    "sentiment_label": "正面/负面/中性",
    "key_events": ["3-5条关键事件描述"],
    "summary": "50字以内的综合情绪总结"
}}"""

    system = "你是专业金融分析师，擅长从新闻中提取关键信息并量化市场情绪。请严格输出 JSON。"
    result_text = _ollama_generate(prompt, system)

    try:
        # 尝试解析 JSON
        start = result_text.find("{")
        end = result_text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(result_text[start:end])
        else:
            raise ValueError("No JSON found")
    except Exception:
        result = {
            "sentiment_score": 0,
            "sentiment_label": "解析失败",
            "key_events": [],
            "summary": "AI 分析异常",
        }

    cache[news_key] = result
    _save_llm_cache(cache)
    return result


def generate_analysis_report(
    fund_name: str,
    code: str,
    indicators: dict,
    news: list[dict],
    sentiment: dict,
    prediction: dict,
) -> str:
    """生成综合 AI 分析报告。"""
    cache = _load_llm_cache()
    cache_key = f"report_{code}_{datetime.now().strftime('%Y%m%d_%H')}"
    if cache_key in cache:
        return cache[cache_key]

    prompt = f"""你是一位资深基金经理和量化分析师。请根据以下信息生成一份简明专业的基金分析报告。

【基金信息】
- 名称：{fund_name}
- 代码：{code}

【技术指标】
- 当前净值：{prediction.get('last_nav', 'N/A')}
- RSI 状态：{indicators.get('rsi_status', 'N/A')}
- 趋势：{indicators.get('trend', 'N/A')}

【ML 模型预测】
- LSTM 趋势：{prediction.get('lstm_trend', 'N/A')}
- 预测变化：{prediction.get('pred_change_pct', 'N/A')}%
- XGBoost 上涨概率：{prediction.get('xgb_up_prob', 0)*100:.1f}%
- 置信度：{prediction.get('confidence_note', 'N/A')}

【新闻消息面】
- 情绪得分：{sentiment.get('sentiment_score', 0)}（{-1}最负面 ~ {+1}最正面）
- 情绪标签：{sentiment.get('sentiment_label', 'N/A')}
- 关键事件：{'; '.join(sentiment.get('key_events', [])[:5])}
- 情绪总结：{sentiment.get('summary', 'N/A')}

请生成 200-300 字的分析报告，包含：
1. 技术面简评（1-2句）
2. 消息面简评（1-2句，结合新闻情绪）
3. 模型预测解读（1-2句）
4. 综合观点（1-2句）
5. 风险提示（1句）

请用中文，语言专业但不晦涩。结尾必须强调「仅供参考，不构成投资建议」。"""

    system = "你是资深基金分析师，专业、客观、谨慎。分析基于数据，不做确定性判断。"
    report = _ollama_generate(prompt, system, timeout=180)

    cache[cache_key] = report
    _save_llm_cache(cache)
    return report


def chat_about_fund(
    fund_name: str,
    code: str,
    question: str,
    context: dict,
) -> str:
    """AI 对话：回答基金相关问题。"""
    prompt = f"""用户正在查看基金「{fund_name}」({code})，并提出问题。

当前数据：
- 最新净值：{context.get('nav', 'N/A')}
- RSI：{context.get('rsi', 'N/A')}
- 趋势：{context.get('trend', 'N/A')}
- LSTM 预测趋势：{context.get('lstm_trend', 'N/A')}
- 涨跌概率：{context.get('up_prob', 'N/A')}

用户问题：{question}

请简洁回答（不超过 200 字），基于数据说话，不做确定性预测。如果问题超出数据范围，诚实说明。结尾加「仅供参考，不构成投资建议」。"""

    system = "你是基金分析助手，回答简洁专业，基于数据，不做确定性预测。"
    return _ollama_generate(prompt, system)


def explain_prediction(prediction: dict, indicators: dict) -> str:
    """用通俗语言解释模型预测逻辑。"""
    cache = _load_llm_cache()
    cache_key = f"explain_{hash(str(prediction))}"
    if cache_key in cache:
        return cache[cache_key]

    prompt = f"""请用通俗易懂的语言（像给朋友讲解一样）解释以下基金预测：

- 当前 RSI：{indicators.get('rsi_status', 'N/A')}
- 当前趋势：{indicators.get('trend', 'N/A')}
- LSTM 模型预测：{prediction.get('lstm_trend', 'N/A')}
- XGBoost 上涨概率：{prediction.get('xgb_up_prob', 0)*100:.1f}%
- 未来{prediction.get('pred_days', 5)}日预测变化：{prediction.get('pred_change_pct', 0)}%

用 100-150 字通俗解释"""
    system = "你用通俗的语言解释技术问题，让非专业人士也能听懂。"
    result = _ollama_generate(prompt, system)

    cache[cache_key] = result
    _save_llm_cache(cache)
    return result
