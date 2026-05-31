"""
ML 预测模块 — LSTM 时间序列预测 + XGBoost 涨跌分类。
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# 特征工程
# ============================================================

FEATURE_COLS = [
    "nav",
    "change_pct",
    "MA5",
    "MA10",
    "MA20",
    "MA60",
    "MACD",
    "MACD_signal",
    "MACD_hist",
    "RSI",
    "BB_upper",
    "BB_middle",
    "BB_lower",
    "vol_MA5",
    "vol_MA20",
    "MA5_bias",      # 价格偏离 MA5 的比例
    "MA20_bias",     # 价格偏离 MA20 的比例
    "vol_ratio",     # 量比
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """从技术指标 DataFrame 构建 ML 特征矩阵。"""
    df = df.copy()

    # 衍生特征
    df["MA5_bias"] = (df["nav"] - df["MA5"]) / df["MA5"].replace(0, np.nan) * 100
    df["MA20_bias"] = (df["nav"] - df["MA20"]) / df["MA20"].replace(0, np.nan) * 100
    df["vol_ratio"] = df["vol_MA5"] / df["vol_MA20"].replace(0, np.nan)

    # 填充缺失值
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)

    return df


def prepare_sequences(
    df: pd.DataFrame, seq_len: int = 60, pred_days: int = 5
) -> tuple[np.ndarray, np.ndarray, "StandardScaler"]:
    """
    从 DataFrame 构建 LSTM 训练序列。

    返回:
        X: (n_samples, seq_len, n_features)
        y: (n_samples, pred_days) — 未来 pred_days 天的净值
        scaler: 已拟合的 StandardScaler
    """
    from sklearn.preprocessing import StandardScaler

    df = build_features(df)

    # 提取特征，跳过前 seq_len 行（不够构建一个序列）
    feature_data = df[FEATURE_COLS].values.astype(np.float32)
    nav_data = df["nav"].values.astype(np.float32)

    scaler = StandardScaler()
    feature_data = scaler.fit_transform(feature_data)

    X, y_list = [], []
    for i in range(len(df) - seq_len - pred_days + 1):
        X.append(feature_data[i : i + seq_len])
        y_list.append(nav_data[i + seq_len : i + seq_len + pred_days])

    if not X:
        return np.array([]), np.array([]), scaler

    return np.array(X), np.array(y_list), scaler


# ============================================================
# LSTM 模型
# ============================================================

class LSTMPredictor:
    """2 层 LSTM + 全连接输出预测未来 N 天净值。"""

    def __init__(self, seq_len=60, n_features=len(FEATURE_COLS), pred_days=5, hidden=64):
        self.seq_len = seq_len
        self.n_features = n_features
        self.pred_days = pred_days
        self.hidden = hidden
        self.model = None
        self.scaler = None

    def _build_model(self):
        import torch
        import torch.nn as nn

        class LSTMNet(nn.Module):
            def __init__(self, n_features, hidden, pred_days):
                super().__init__()
                self.lstm1 = nn.LSTM(n_features, hidden, batch_first=True)
                self.dropout1 = nn.Dropout(0.2)
                self.lstm2 = nn.LSTM(hidden, hidden // 2, batch_first=True)
                self.dropout2 = nn.Dropout(0.2)
                self.fc = nn.Linear(hidden // 2, pred_days)

            def forward(self, x):
                x, _ = self.lstm1(x)
                x = self.dropout1(x)
                x, _ = self.lstm2(x)
                x = self.dropout2(x)
                x = x[:, -1, :]  # 取最后一个时间步
                return self.fc(x)

        self.model = LSTMNet(self.n_features, self.hidden, self.pred_days)

    def train(self, df: pd.DataFrame, epochs: int = 50, verbose: bool = False):
        """训练 LSTM 模型。"""
        import torch

        X, y, self.scaler = prepare_sequences(df, self.seq_len, self.pred_days)
        if len(X) < 10:
            raise ValueError(f"数据量不足（需要至少 {self.seq_len + self.pred_days + 10} 条，当前 {len(df)} 条）")

        self._build_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        y_t = torch.tensor(y, dtype=torch.float32).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = torch.nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = self.model(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  LSTM Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """预测未来 pred_days 天的净值。返回 shape (pred_days,) 的数组。"""
        import torch

        if self.model is None:
            raise RuntimeError("模型未训练，请先调用 train()")

        df_feat = build_features(df)
        feature_data = df_feat[FEATURE_COLS].values.astype(np.float32)

        if self.scaler is None:
            raise RuntimeError("Scaler 未初始化")
        feature_data = self.scaler.transform(feature_data)

        # 取最后 seq_len 条
        if len(feature_data) < self.seq_len:
            raise ValueError(f"数据不足：需要至少 {self.seq_len} 条，当前 {len(feature_data)} 条")

        seq = feature_data[-self.seq_len:]
        seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        device = next(self.model.parameters()).device
        seq_t = seq_t.to(device)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(seq_t).cpu().numpy().flatten()

        return pred

    def save(self, code: str):
        path = os.path.join(MODEL_DIR, f"{code}_lstm.pkl")
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "seq_len": self.seq_len,
                    "n_features": self.n_features,
                    "pred_days": self.pred_days,
                    "hidden": self.hidden,
                    "model_state": self.model.state_dict() if self.model else None,
                    "scaler": self.scaler,
                },
                f,
            )

    def load(self, code: str) -> bool:
        path = os.path.join(MODEL_DIR, f"{code}_lstm.pkl")
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.seq_len = data["seq_len"]
        self.n_features = data["n_features"]
        self.pred_days = data["pred_days"]
        self.hidden = data["hidden"]
        self.scaler = data["scaler"]
        self._build_model()
        if data["model_state"]:
            self.model.load_state_dict(data["model_state"])
        return True


# ============================================================
# XGBoost 分类器
# ============================================================

class XGBoostClassifier:
    """预测次日涨跌的二分类模型。"""

    def __init__(self):
        self.model = None
        self.scaler = None

    def train(self, df: pd.DataFrame):
        """训练 XGBoost 分类器。"""
        import xgboost as xgb
        from sklearn.preprocessing import StandardScaler

        df = build_features(df)

        # 标签: 次日涨 (1) / 跌 (0)
        df["target"] = (df["change_pct"].shift(-1) > 0).astype(int)
        df = df.dropna(subset=["target"])

        feature_data = df[FEATURE_COLS].values.astype(np.float32)
        y = df["target"].values.astype(int)

        if len(y) < 50:
            raise ValueError(f"数据量不足（至少需要 50 条，当前 {len(y)} 条）")

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(feature_data)

        # 计算类别权重（处理样本不平衡）
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        scale_pos_weight = n_neg / max(n_pos, 1)

        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        )
        self.model.fit(X, y)

    def predict_proba(self, df: pd.DataFrame) -> float:
        """返回明日上涨概率 (0~1)。"""
        if self.model is None or self.scaler is None:
            raise RuntimeError("模型未训练，请先调用 train()")

        df_feat = build_features(df)
        feature_data = df_feat[FEATURE_COLS].values.astype(np.float32)
        X = self.scaler.transform(feature_data)

        # 取最后一条
        last = X[-1:]

        proba = self.model.predict_proba(last)[0]
        # proba[0] = 跌的概率, proba[1] = 涨的概率
        return float(proba[1])

    def save(self, code: str):
        path = os.path.join(MODEL_DIR, f"{code}_xgb.pkl")
        with open(path, "wb") as f:
            pickle.dump(
                {"model": self.model, "scaler": self.scaler},
                f,
            )

    def load(self, code: str) -> bool:
        path = os.path.join(MODEL_DIR, f"{code}_xgb.pkl")
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        return True


# ============================================================
# 统一预测接口
# ============================================================

def train_and_predict(
    df: pd.DataFrame,
    code: str,
    pred_days: int = 5,
    force_retrain: bool = False,
) -> dict:
    """
    训练（或加载）模型并返回预测结果。

    返回:
        {
            "lstm_pred": [未来N天净值预测列表],
            "lstm_dates": [预测日期列表],
            "xgb_up_prob": 明日上涨概率,
            "lstm_trend": "看涨" / "看跌" / "震荡",
            "confidence_note": 置信度说明,
        }
    """
    from indicators import compute_all

    df = compute_all(df)

    # ---- LSTM ----
    lstm = LSTMPredictor(pred_days=pred_days)
    if not force_retrain and lstm.load(code):
        pass  # 已加载缓存模型
    else:
        lstm.train(df, epochs=50)
        lstm.save(code)

    # ---- XGBoost ----
    xgb_clf = XGBoostClassifier()
    if not force_retrain and xgb_clf.load(code):
        pass
    else:
        xgb_clf.train(df)
        xgb_clf.save(code)

    # ---- 执行预测 ----
    lstm_pred = lstm.predict(df)
    up_prob = xgb_clf.predict_proba(df)

    # 生成预测日期（跳过周末，简化处理）
    last_date = df["date"].iloc[-1]
    from datetime import timedelta

    future_dates = []
    d = last_date + timedelta(days=1)
    while len(future_dates) < pred_days:
        if d.weekday() < 5:  # 周一到周五
            future_dates.append(d)
        d += timedelta(days=1)

    # 趋势判断
    last_nav = df["nav"].iloc[-1]
    lstm_trend = "看涨" if lstm_pred[-1] > last_nav else "看跌"
    if abs(lstm_pred[-1] - last_nav) / last_nav < 0.005:
        lstm_trend = "震荡"

    # 置信度
    if up_prob > 0.7 or up_prob < 0.3:
        confidence = "较高"
    elif up_prob > 0.55 or up_prob < 0.45:
        confidence = "中等"
    else:
        confidence = "较低（市场方向不明确）"

    return {
        "lstm_pred": [round(float(v), 4) for v in lstm_pred],
        "lstm_dates": [d.strftime("%Y-%m-%d") for d in future_dates],
        "xgb_up_prob": round(up_prob, 4),
        "lstm_trend": lstm_trend,
        "confidence_note": confidence,
        "last_nav": round(float(last_nav), 4),
        "pred_change_pct": round(float((lstm_pred[-1] - last_nav) / last_nav * 100), 2),
    }
