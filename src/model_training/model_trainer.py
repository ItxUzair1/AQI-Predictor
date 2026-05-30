import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
import joblib
import json
from datetime import datetime, timedelta

sys.path.append(os.getcwd())
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger("Model Trainer")


class ModelTrainer:
    def __init__(self, city_name="karachi"):
        self.city_name = city_name
        self.model_name = f"aqi_predictor_{city_name}"
        self.artifacts_dir = "artifacts"
        self.local_model_path = os.path.join(self.artifacts_dir, "model.joblib")
        self.feature_names_path = os.path.join(self.artifacts_dir, "feature_names.json")
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def _get_hopsworks_project(self):
        """Logs in to Hopsworks and returns the project handle."""
        # pyrefly: ignore [missing-import]
        import hopsworks
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv("HOPSWORKS_API_KEY")
        project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "AQI_Prediction_System_10")
        host = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")

        if not api_key:
            raise CustomException("HOPSWORKS_API_KEY not found in environment variables", sys)

        # Windows workaround for Hopsworks certs
        if sys.platform == "win32":
            local_tmp = os.path.join(os.getcwd(), "tmp")
            os.makedirs(local_tmp, exist_ok=True)
            os.environ["TMP"] = local_tmp
            os.environ["TEMP"] = local_tmp

        project = hopsworks.login(
            host=host,
            project=project_name,
            api_key_value=api_key
        )
        return project

    def fetch_data(self):
        """
        Fetches historical AQI data from the Hopsworks feature store.
        Returns a DataFrame sorted by ingestion_timestamp.
        Falls back to the online store if the offline store (Hudi) isn't ready.
        """
        try:
            logger.info("Connecting to Hopsworks feature store...")
            project = self._get_hopsworks_project()
            fs = project.get_feature_store()

            aqi_fg = fs.get_feature_group(name="aqi_features", version=6)

            # Try offline store first (default), fall back to online store, then local CSV cache
            try:
                logger.info("Reading from offline store...")
                df = aqi_fg.select_all().read()
            except Exception as offline_err:
                logger.warning(f"Offline store read failed: {offline_err}")
                logger.info("Checking local offline cache (data/aqi_features.csv)...")
                local_path = os.path.join("data", "aqi_features.csv")
                if os.path.exists(local_path):
                    try:
                        df = pd.read_csv(local_path)
                        logger.info(f"Successfully loaded {len(df)} rows from local offline cache.")
                    except Exception as cache_err:
                        logger.warning(f"Failed to read local cache: {cache_err}")
                        df = pd.DataFrame()
                else:
                    df = pd.DataFrame()
                
                if df.empty:
                    logger.info("Falling back to online store read...")
                    try:
                        df = aqi_fg.select_all().read(online=True)
                    except Exception as online_err:
                        logger.error(f"Online store read also failed: {online_err}")
                        logger.info("Hopsworks cluster is out of resources. Generating fallback data...")
                        sys.path.append(os.getcwd())
                        from scripts.backfill_data import generate_historical_data
                        df = generate_historical_data(self.city_name, days=45)
                        # Save it locally so next time it is cached
                        os.makedirs("data", exist_ok=True)
                        df.to_csv(local_path, index=False)
                        return df, project

            # Filter to the target city
            if 'city' in df.columns:
                df = df[df['city'].str.lower().str.contains(self.city_name.lower())]

            # Sort chronologically — CRITICAL for time-based shifting
            # NOTE: We use ingestion_timestamp (set at pipeline runtime) NOT the API 'timestamp'
            # because the AQICN station's timestamp field is frozen/stale (stuck at 2025-03-04).
            df['ingestion_timestamp'] = pd.to_datetime(df['ingestion_timestamp'])
            df = df.sort_values(by="ingestion_timestamp").reset_index(drop=True)

            logger.info(f"Fetched {len(df)} rows for city: {self.city_name}")
            return df, project

        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise CustomException(e, sys)

    def _add_lag_features(self, df):
        """
        Adds lag and rolling-window features so the model can capture
        temporal trends rather than relying on the raw current AQI value.

        Features created:
          - aqi_lag_24h        : AQI value 24 hours ago
          - aqi_lag_48h        : AQI value 48 hours ago
          - aqi_lag_72h        : AQI value 72 hours ago
          - aqi_rolling_mean_24h : Rolling mean AQI over the past 24 hours
          - aqi_rolling_std_24h  : Rolling std AQI over the past 24 hours
          - aqi_rolling_mean_72h : Rolling mean AQI over the past 72 hours
          - aqi_diff_24h       : AQI change over the last 24 hours (trend direction)
          - pm25_rolling_mean_24h: Rolling mean PM2.5 over the past 24 hours
          - wind_speed_rolling_mean_24h: Rolling mean wind speed over the past 24 hours
        """
        df = df.copy()
        df = df.sort_values('ingestion_timestamp').reset_index(drop=True)

        # Lag features (assuming hourly data)
        df['aqi_lag_24h'] = df['aqi'].shift(24)
        df['aqi_lag_48h'] = df['aqi'].shift(48)
        df['aqi_lag_72h'] = df['aqi'].shift(72)

        # Rolling statistics
        df['aqi_rolling_mean_24h'] = df['aqi'].rolling(window=24, min_periods=12).mean()
        df['aqi_rolling_std_24h'] = df['aqi'].rolling(window=24, min_periods=12).std()
        df['aqi_rolling_mean_72h'] = df['aqi'].rolling(window=72, min_periods=36).mean()

        # Trend feature: how much AQI changed in the last 24h
        df['aqi_diff_24h'] = df['aqi'] - df['aqi'].shift(24)

        # Rolling features for other important signals
        df['pm25_rolling_mean_24h'] = df['pm25'].rolling(window=24, min_periods=12).mean()
        df['wind_speed_rolling_mean_24h'] = df['wind_speed'].rolling(window=24, min_periods=12).mean()

        logger.info("Added lag and rolling features")
        return df

    def prepare_data(self, df):
        """
        Prepares features (X) and targets (y) for multi-day AQI prediction.

        Strategy: For each row at time T, we create 3 target columns:
          - target_day1: AQI at T + 1 day  (tomorrow)
          - target_day2: AQI at T + 2 days (day after tomorrow)
          - target_day3: AQI at T + 3 days

        Uses shift-based approach: assuming roughly hourly data,
        shift by 24/48/72 rows to get 1/2/3-day-ahead targets.

        IMPORTANT: The current 'aqi' column is EXCLUDED from features to prevent
        data leakage. Instead, lag and rolling features capture the temporal signal.
        """
        try:
            logger.info("Preparing data: creating 3-day forward targets (day1, day2, day3)...")

            df = df.copy()

            # Deduplicate on ingestion_timestamp to avoid issues
            df = df.drop_duplicates(subset=['ingestion_timestamp']).reset_index(drop=True)
            logger.info(f"Rows after dedup: {len(df)}")

            # ── Step 1: Add lag/rolling features BEFORE creating the targets ──
            df = self._add_lag_features(df)

            # ── Step 2: Build targets using shift ────────────────────────────
            # Determine the actual cadence (median gap between rows)
            time_diffs = df['ingestion_timestamp'].diff().dropna()
            if len(time_diffs) > 0:
                median_gap_hours = time_diffs.median().total_seconds() / 3600
                logger.info(f"Median gap between rows: {median_gap_hours:.1f} hours")
                rows_per_day = max(1, int(round(24 / max(median_gap_hours, 0.1))))
            else:
                rows_per_day = 24  # default: assume hourly
            logger.info(f"Using {rows_per_day} rows per day for shift-based targets")

            target_cols = []
            for horizon_days in [1, 2, 3]:
                col_name = f'target_day{horizon_days}'
                shift_amount = rows_per_day * horizon_days
                # Shift backwards (negative) means "future value at row + shift_amount"
                df[col_name] = df['aqi'].shift(-shift_amount)
                target_cols.append(col_name)

            # Drop rows that could not find ALL three targets
            df = df.dropna(subset=target_cols)

            logger.info(f"Rows available for training after multi-day target creation: {len(df)}")

            if len(df) < 50:
                logger.warning(
                    f"Only {len(df)} labelled rows available. "
                    "The model may not perform well. Ensure you have at least 3+ days of historical data."
                )

            # ── Step 3: Define feature columns ───────────────────────────────
            # EXCLUDE 'aqi' to prevent leakage — the model should NOT see the
            # current AQI value directly. The lag/rolling features capture the
            # temporal signal without leaking the answer.
            non_feature_cols = {
                'city', 'timestamp', 'target_day', 'target_aqi',
                'ingestion_timestamp', 'target',
                'target_day1', 'target_day2', 'target_day3',
                'aqi'  # ← KEY: remove current AQI to prevent leakage
            }
            feature_cols = [c for c in df.columns if c not in non_feature_cols]

            # Drop rows with NaN in feature columns (from lag features at the start)
            df = df.dropna(subset=feature_cols)

            logger.info(f"Rows after dropping NaN lag rows: {len(df)}")
            logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

            X = df[feature_cols]
            y = df[target_cols]  # DataFrame with 3 columns

            # Persist feature column names so the app can align inference features
            with open(self.feature_names_path, 'w') as f:
                json.dump(feature_cols, f)
            logger.info(f"Feature names saved to {self.feature_names_path}")

            return X, y

        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise CustomException(e, sys)

    def train_and_register(self):
        """
        Full workflow:
          1. Fetch data from feature store.
          2. Prepare 3-day-ahead targets.
          3. Train multiple models and select the best.
          4. Evaluate.
          5. Register in Hopsworks Model Registry.
        """
        try:
            df, project = self.fetch_data()

            if df.empty:
                logger.error("No data found in feature store. Cannot train.")
                return None

            X, y = self.prepare_data(df)

            if len(X) == 0:
                logger.error("No labelled samples after target creation. Cannot train.")
                return None

            # Chronological split — never shuffle time-series data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            logger.info(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows...")

            # Sanity check: log target statistics
            # Logging disabled here due to multi-output target format
            # logger.info(f"Target stats — train  mean: {y_train.mean():.1f}, std: {y_train.std():.1f}")
            # logger.info(f"Target stats — test  mean: {y_test.mean():.1f}, std: {y_test.std():.1f}")

            # Define candidate models
            # XGBoost and GradientBoosting need MultiOutputRegressor wrappers;
            # RandomForest and LinearRegression support multi-output natively.
            candidate_models = {
                "XGBoost": MultiOutputRegressor(
                    xgb.XGBRegressor(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=5,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                    )
                ),
                "RandomForest": RandomForestRegressor(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42
                ),
                "GradientBoosting": MultiOutputRegressor(
                    GradientBoostingRegressor(
                        n_estimators=150,
                        learning_rate=0.05,
                        max_depth=4,
                        random_state=42
                    )
                ),
                "LinearRegression": LinearRegression()
            }

            best_model_name = None
            best_model = None
            best_mae = float('inf')
            best_metrics = {}

            for name, candidate_model in candidate_models.items():
                logger.info(f"Training {name} model (multi-output: day1, day2, day3)...")
                candidate_model.fit(X_train, y_train)

                # Evaluate — predictions shape is (n_samples, 3)
                predictions = candidate_model.predict(X_test)
                mae = float(mean_absolute_error(y_test, predictions))
                rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

                # Per-day MAE for logging
                for i, day_label in enumerate(['Day1', 'Day2', 'Day3']):
                    day_mae = float(mean_absolute_error(y_test.iloc[:, i], predictions[:, i]))
                    logger.info(f"  {name} {day_label} MAE: {day_mae:.2f}")

                logger.info(f"{name} Evaluation (avg) → MAE: {mae:.2f}  |  RMSE: {rmse:.2f}")

                # Sanity check: are predictions varying?
                pred_std = float(np.std(predictions))
                logger.info(f"  Prediction std: {pred_std:.2f} (should NOT be near 0)")

                if mae < best_mae:
                    best_mae = mae
                    best_model_name = name
                    best_model = candidate_model
                    best_metrics = {"mae": mae, "rmse": rmse}

            logger.info(f"Selected Best Model: {best_model_name} with MAE: {best_mae:.2f}")

            # Save best model and feature names locally
            joblib.dump(best_model, self.local_model_path)
            logger.info(f"Best model saved locally to: {self.local_model_path}")

            # --- Push to Hopsworks Model Registry ---
            logger.info("Registering best model in Hopsworks Model Registry...")
            mr = project.get_model_registry()

            aqi_model = mr.python.create_model(
                name=self.model_name,
                metrics=best_metrics,
                description=(
                    f"{best_model_name} multi-output model predicting AQI for next 3 days for {self.city_name}. "
                    f"Trained on {len(X_train)} samples. Outputs: [day1, day2, day3]. "
                    f"Features: lag/rolling AQI, pollutants, weather, time."
                )
            )
            # Save both the model binary and the feature names list into the registry
            aqi_model.save(self.artifacts_dir)
            logger.info(f"Model v{aqi_model.version} ({best_model_name}) registered: {self.model_name}")

            return aqi_model

        except Exception as e:
            logger.error(f"Model training/registration failed: {str(e)}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    trainer = ModelTrainer("karachi")
    trainer.train_and_register()
