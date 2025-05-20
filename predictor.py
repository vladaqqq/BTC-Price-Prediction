import pandas as pd
import numpy as np
import requests
from typing import List, Tuple, Dict
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from lightgbm import LGBMRegressor
import optuna
from optuna.samplers import TPESampler
import logging

logger = logging.getLogger(__name__)

class EnhancedBTCPredictor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.model = None
        self.best_params = None
        self.initialized = False
        self.feature_names = []
        self.scaler_fitted = False
        self.metrics: Dict[str, float] = {}
        self.random_seed = 42

    def initialize(self):
        try:
            logger.info("Initializing model with metrics...")
            df = self.load_data(days=3000)
            df = self.enhanced_feature_engineering(df)
            X, y = self.prepare_features(df)

            self.scaler.fit(X)
            self.scaler_fitted = True

            X_scaled = self.scaler.transform(X)
            self.best_params = self.optimize_hyperparameters(
                pd.DataFrame(X_scaled, columns=X.columns), y
            )

            self.train_final_model(X, y)
            self.calculate_metrics(X, y)

            self.initialized = True
            logger.info(f"Model initialized. Metrics: {self.metrics}")
        except Exception as e:
            logger.critical(f"Initialization failed: {str(e)}")
            raise

    def load_data(self, days: int = 2000) -> pd.DataFrame:
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
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['volume_ma'] = df['volume'].rolling(7).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        windows = [7, 14, 30, 50]
        for window in windows:
            df[f'ma_{window}'] = df['close'].rolling(window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            df[f'volatility_{window}'] = df['close'].pct_change().rolling(window).std()

        for lag in [1, 3, 7, 14]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)

        df['target'] = (df['close'].shift(-1) - df['close']) / df['close']
        return df.dropna().copy()

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        self.feature_names = df.drop(columns=['target']).columns.tolist()
        return df.drop(columns=['target']), df['target']

    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> dict:
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'random_state': self.random_seed
            }

            tscv = TimeSeriesSplit(n_splits=5)
            scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = LGBMRegressor(**params)
                model.fit(X_train, y_train, feature_name=X.columns.tolist(), categorical_feature=[])
                preds = model.predict(X_val)
                scores.append(mean_absolute_error(y_val, preds))

            return np.mean(scores)

        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=self.random_seed))
        study.optimize(objective, n_trials=30)
        return study.best_params

    def train_final_model(self, X: pd.DataFrame, y: pd.Series):
        X_scaled = self.scaler.transform(X)
        self.model = LGBMRegressor(**self.best_params, random_state=self.random_seed)
        self.model.fit(X_scaled, y, feature_name=X.columns.tolist(), categorical_feature=[])

    def calculate_metrics(self, X: pd.DataFrame, y: pd.Series):
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        mae = mean_absolute_error(y, y_pred)
        directional_accuracy = np.mean(np.sign(y) == np.sign(y_pred)) * 100
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        self.metrics = {
            "MAE": round(mae, 6),
            "MAE (%)": round(mae * 100, 2),
            "Directional Accuracy (%)": round(directional_accuracy, 2),
            "R²": round(r2, 4),
            "RMSE": round(rmse, 6)
        }

    def predict_multiple_days(self, X: pd.DataFrame, days: List[int]) -> dict:
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

# Экземпляр модели
predictor = EnhancedBTCPredictor()
