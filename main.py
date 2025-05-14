from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import logging
import pandas as pd
import numpy as np
import requests
from typing import List, Tuple, Dict
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_squared_error
)
import optuna
from optuna.samplers import TPESampler

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="BTC Price Predictor Pro", version="4.3")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class EnhancedBTCPredictor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.model = None
        self.best_params = None
        self.initialized = False
        self.feature_names = []
        self.scaler_fitted = False
        self.metrics: Dict[str, float] = {}
        self.random_seed = 42  # Фиксированный seed для воспроизводимости

    def initialize(self):
        """Инициализация модели с подсчётом метрик"""
        try:
            logger.info("Initializing model with metrics...")
            df = self.load_data(days=3000)
            df = self.enhanced_feature_engineering(df)
            X, y = self.prepare_features(df)

            self.scaler.fit(X)
            self.scaler_fitted = True

            # Оптимизация гиперпараметров
            X_scaled = self.scaler.transform(X)
            self.best_params = self.optimize_hyperparameters(
                pd.DataFrame(X_scaled, columns=X.columns), y
            )

            # Обучение и оценка
            self.train_final_model(X, y)
            self.calculate_metrics(X, y)

            self.initialized = True
            logger.info(f"Model initialized. Metrics: {self.metrics}")

        except Exception as e:
            logger.critical(f"Initialization failed: {str(e)}")
            raise

    def calculate_metrics(self, X: pd.DataFrame, y: pd.Series):
        """Вычисление всех метрик качества модели"""
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)

        # MAE
        mae = mean_absolute_error(y, y_pred)

        # Directional Accuracy
        correct_direction = np.sign(y) == np.sign(y_pred)
        directional_accuracy = np.mean(correct_direction) * 100

        # R²
        r2 = r2_score(y, y_pred)

        # RMSE
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        self.metrics = {
            "MAE": round(mae, 6),
            "MAE (%)": round(mae * 100, 2),
            "Directional Accuracy (%)": round(directional_accuracy, 2),
            "R²": round(r2, 4),
            "RMSE": round(rmse, 6)
        }

    def load_data(self, days: int = 2000) -> pd.DataFrame:
        """Загрузка данных с Binance API"""
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": "BTCUSDT",
            "interval": "1d",
            "limit": min(days, 3000)
        }
        response = requests.get(url, params=params)
        response.raise_for_status()

        df = pd.DataFrame(
            response.json(),
            columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades_count",
                "taker_buy_volume", "taker_quote_volume", "ignore"
            ]
        )

        numeric_cols = ["open", "high", "low", "close", "volume"]
        df[numeric_cols] = df[numeric_cols].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df.set_index("timestamp").sort_index().ffill()

    def enhanced_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Генерация признаков"""
        # Ценовые фичи
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['price_range'] = (df['high'] - df['low']) / df['close']

        # Объемные индикаторы
        df['volume_ma'] = df['volume'].rolling(7).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        # Скользящие статистики
        windows = [7, 14, 30, 50]
        for window in windows:
            df[f'ma_{window}'] = df['close'].rolling(window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            df[f'volatility_{window}'] = df['close'].pct_change().rolling(window).std()

        # Лаговые признаки
        for lag in [1, 3, 7, 14]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)

        # Целевая переменная
        df['target'] = (df['close'].shift(-1) - df['close']) / df['close']

        return df.dropna().copy()

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Подготовка данных"""
        self.feature_names = df.drop(columns=['target']).columns.tolist()
        return df.drop(columns=['target']), df['target']

    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Оптимизация гиперпараметров с фиксированным seed"""

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'random_state': self.random_seed  # Фиксируем seed
            }

            tscv = TimeSeriesSplit(n_splits=5)
            scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = LGBMRegressor(**params)
                model.fit(
                    X_train, y_train,
                    feature_name=X.columns.tolist(),
                    categorical_feature=[]
                )
                preds = model.predict(X_val)
                scores.append(mean_absolute_error(y_val, preds))

            return np.mean(scores)

        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_seed)  # Фиксируем seed
        )
        study.optimize(objective, n_trials=30)
        return study.best_params

    def train_final_model(self, X: pd.DataFrame, y: pd.Series):
        """Обучение финальной модели с фиксированным seed"""
        X_scaled = self.scaler.transform(X)
        self.model = LGBMRegressor(
            **self.best_params,
            random_state=self.random_seed  # Фиксируем seed
        )
        self.model.fit(
            X_scaled, y,
            feature_name=X.columns.tolist(),
            categorical_feature=[]
        )

    def predict_multiple_days(self, X: pd.DataFrame, days: List[int]) -> dict:
        """Прогнозирование с выводом метрик"""
        if not self.initialized:
            raise RuntimeError("Model not initialized")

        predictions = {}
        current_price = X['close'].iloc[-1]
        X_scaled = self.scaler.transform(X)
        X_final = pd.DataFrame(X_scaled, columns=self.feature_names)

        for days_ahead in days:
            pred_return = self.model.predict(X_final)[0]
            predicted_price = current_price * (1 + pred_return) ** days_ahead
            predictions[f'{days_ahead}_days'] = {
                'price': round(predicted_price, 2),
                'return (%)': round(pred_return * 100, 2),
                'confidence': f"±{self.metrics['MAE (%)']}% (DA: {self.metrics['Directional Accuracy (%)']}%)"
            }

        return {
            "current_price": round(current_price, 2),
            "metrics": self.metrics,
            "predictions": predictions
        }


# Инициализация
predictor = EnhancedBTCPredictor()


@app.on_event("startup")
async def startup():
    predictor.initialize()


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/predict")
async def predict(days: str = "1,3,7,30,90,180"):
    try:
        days_list = [int(d) for d in days.split(",")]
        df = predictor.load_data(days=60)
        df = predictor.enhanced_feature_engineering(df)
        X = df.drop(columns=['target'], errors='ignore').tail(1)

        result = predictor.predict_multiple_days(X, days_list)
        return {"status": "success", **result}

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9331)