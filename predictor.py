import os
import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMRegressor
from scipy.stats import zscore
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import optuna
from config import logger


class BTCPredictor:
    def __init__(self, random_seed: int = 42, window_size: int = 5):
        self.scaler = RobustScaler()
        self.model = None
        self.metrics = {}
        self.random_seed = random_seed
        self.window_size = window_size
        self.initialized = False

    def initialize(self):
        try:
            os.makedirs("static", exist_ok=True)
            os.makedirs("templates", exist_ok=True)

            df = self.load_data().pipe(self.feature_engineering)
            X, y = df.drop(columns=['target']), df['target']
            X = X.mask(np.abs(zscore(X)) > 3).ffill().bfill()
            self.scaler.fit(X)

            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=self.random_seed)
            )
            study.optimize(lambda trial: self.objective(trial, X, y), n_trials=40)

            # Обучаем финальную модель на DataFrame
            self.model = LGBMRegressor(**study.best_params, random_state=self.random_seed)
            X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
            self.model.fit(X_scaled, y)

            self.calculate_metrics(X, y)
            logger.info(f"Model metrics: {self.metrics}")
            self.initialized = True
            logger.info("Model initialized successfully")

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.initialized = False

    def load_data(self) -> pd.DataFrame:
        data = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "1d", "limit": 3000}
        ).json()
        df = pd.DataFrame(data)[[0,1,2,3,4,5]].astype(float)
        df.columns = ['timestamp','open','high','low','close','volume']
        return df.set_index(pd.to_datetime(df['timestamp'], unit='ms'))

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(
            log_ret=np.log(df['close'] / df['close'].shift(1)),
            rsi=lambda x: 100 - (100 / (
                1 + (x.close.diff().clip(lower=0).rolling(14).mean() /
                     x.close.diff().clip(upper=0).abs().rolling(14).mean())
            )),
            macd=lambda x: x.close.ewm(span=12).mean() - x.close.ewm(span=26).mean(),
            obv=lambda x: (np.sign(x.close.diff()) * x.volume).cumsum(),
            **{f'close_lag_{lag}': df.close.shift(lag) for lag in [1,3,7,14]},
            target=lambda x: (x.close.shift(-1) - x.close) / x.close
        ).dropna()
        return df

    def objective(self, trial, X: pd.DataFrame, y: pd.Series) -> float:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 800, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0)
        }
        scores = []
        tscv = TimeSeriesSplit(n_splits=5)
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = LGBMRegressor(**params)
            # внутренняя CV: обучение и предсказание на numpy — ок
            model.fit(self.scaler.transform(X_train), y_train)
            preds = model.predict(self.scaler.transform(X_val))
            scores.append(np.mean(np.sign(preds) == np.sign(y_val)))

        return float(np.mean(scores))

    def calculate_metrics(self, X: pd.DataFrame, y: pd.Series):
        # Оборачиваем в DataFrame, чтобы сохранить имена признаков
        X_df = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        preds = self.model.predict(X_df)
        mae = mean_absolute_error(y, preds)
        r2 = r2_score(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        dir_acc = np.mean(np.sign(y) == np.sign(preds)) * 100
        acc7 = np.mean(np.sign(y.iloc[-7:]) == np.sign(preds[-7:])) * 100

        self.metrics = {
            "MAE": f"{mae:.6f}",
            "R²": f"{r2:.4f}",
            "RMSE": f"{rmse:.6f}",
            "Direction Accuracy": f"{dir_acc:.1f}%",
            "7-Day Accuracy": f"{acc7:.1f}%"
        }

    def predict(self, days: list[int]) -> dict:
        df = self.load_data().pipe(self.feature_engineering)
        X = df.drop(columns='target').tail(self.window_size)
        # Оборачиваем в DataFrame для predict
        X_df = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        preds = self.model.predict(X_df)

        avg_ret = np.mean(preds)
        cur_price = df.close.iloc[-1]

        preds_dict = {}
        for d in days:
            if d in (1, 3, 7):
                price = round(cur_price * (1 + avg_ret) ** d, 2)
                preds_dict[f"{d}_days"] = {
                    "price": price,
                    "return (%)": round(avg_ret * 100, 2)
                }

        return {
            "current_price": round(cur_price, 2),
            "predictions": preds_dict,
            "metrics": self.metrics
        }