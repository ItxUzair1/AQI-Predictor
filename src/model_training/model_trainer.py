import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
        import hopsworks
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv("HOPSWORKS_API_KEY")
        project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "AQI_Prediction_ML_System")
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
        """
        try:
            logger.info("Connecting to Hopsworks feature store...")
            project = self._get_hopsworks_project()
            fs = project.get_feature_store()

            aqi_fg = fs.get_feature_group(name="aqi_features", version=2)
            df = aqi_fg.select_all().read()

            # Filter to the target city
            if 'city' in df.columns:
                df = df[df['city'] == self.city_name]

            # Sort chronologically — CRITICAL for time-based shifting
            df['ingestion_timestamp'] = pd.to_datetime(df['ingestion_timestamp'])
            df = df.sort_values(by="ingestion_timestamp").reset_index(drop=True)

            logger.info(f"Fetched {len(df)} rows for city: {self.city_name}")
            return df, project

        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise CustomException(e, sys)

    def prepare_data(self, df):
        """
        Prepares features (X) and target (y) for 3-day AQI prediction.

        Strategy: For each row at time T, the target is the AQI value at T + 3 days.
        We use a time-aware merge (pd.merge_asof) instead of a positional shift,
        so the result is correct even when the historical data is sparse or irregular.
        """
        try:
            logger.info("Preparing data: creating 3-day forward target...")

            df = df.copy()

            # Build a lookup table: for each row, what was the AQI 3 days later?
            future = df[['ingestion_timestamp', 'aqi']].copy()
            future['ingestion_timestamp'] = future['ingestion_timestamp'] - timedelta(days=3)
            future = future.rename(columns={'aqi': 'target'})

            # Merge: for each row in df, find the closest future reading ~3 days later
            # tolerance = ±6 hours to account for missing hours
            df = pd.merge_asof(
                df.sort_values('ingestion_timestamp'),
                future.sort_values('ingestion_timestamp'),
                on='ingestion_timestamp',
                direction='nearest',
                tolerance=pd.Timedelta(hours=6)
            )

            # Drop rows that could not find a target within tolerance
            df = df.dropna(subset=['target'])

            logger.info(f"Rows available for training after 3-day target creation: {len(df)}")

            if len(df) < 50:
                logger.warning(
                    f"Only {len(df)} labelled rows available. "
                    "The model may not perform well. Ensure you have at least 3+ days of historical data."
                )

            # Columns to exclude from features
            non_feature_cols = [
                'city', 'timestamp', 'target_day', 'target_aqi',
                'ingestion_timestamp', 'target'
            ]
            feature_cols = [c for c in df.columns if c not in non_feature_cols]

            X = df[feature_cols]
            y = df['target']

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
          3. Train XGBoost.
          4. Evaluate.
          5. Register in Hopsworks Model Registry (always pushes; registry versions automatically).
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

            model = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                early_stopping_rounds=20,
                eval_metric="mae"
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )

            # Evaluate
            predictions = model.predict(X_test)
            mae = float(mean_absolute_error(y_test, predictions))
            rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
            logger.info(f"Model Evaluation → MAE: {mae:.2f}  |  RMSE: {rmse:.2f}")

            # Save model and feature names locally
            joblib.dump(model, self.local_model_path)
            logger.info(f"Model saved locally to: {self.local_model_path}")

            # --- Push to Hopsworks Model Registry ---
            logger.info("Registering model in Hopsworks Model Registry...")
            mr = project.get_model_registry()

            aqi_model = mr.python.create_model(
                name=self.model_name,
                metrics={"mae": mae, "rmse": rmse},
                description=(
                    f"XGBoost model predicting AQI 3 days ahead for {self.city_name}. "
                    f"Trained on {len(X_train)} samples."
                )
            )
            # Save both the model binary and the feature names list into the registry
            aqi_model.save(self.artifacts_dir)
            logger.info(f"Model v{aqi_model.version} registered: {self.model_name}")

            return aqi_model

        except Exception as e:
            logger.error(f"Model training/registration failed: {str(e)}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    trainer = ModelTrainer("karachi")
    trainer.train_and_register()
