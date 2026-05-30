import os
# pyrefly: ignore [missing-import]
import hopsworks
import pandas as pd
from src.logger import get_logger
from src.exception import CustomException
import sys
from dotenv import load_dotenv

load_dotenv()
logger = get_logger("Feature Store Ingestion")

class FeatureStoreIngestion:
    def __init__(self):
        self.api_key = os.getenv("HOPSWORKS_API_KEY")
        self.project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "AQI_Prediction_System_10")

    def get_feature_store(self):
        """
        Connects to Hopsworks and returns the feature store handle.
        """
        try:
            if not self.api_key:
                raise CustomException("HOPSWORKS_API_KEY not found in environment variables", sys)
            
            host = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
            
            # Windows workaround for Hopsworks certs in /tmp
            if sys.platform == "win32":
                local_tmp = os.path.join(os.getcwd(), "tmp")
                os.makedirs(local_tmp, exist_ok=True)
                os.environ["TMP"] = local_tmp
                os.environ["TEMP"] = local_tmp
                # Also try to create /tmp at the root of the current drive if possible
                try:
                    if not os.path.exists("/tmp"):
                        os.makedirs("/tmp", exist_ok=True)
                except:
                    pass

            project = hopsworks.login(
                host=host,
                project=self.project_name,
                api_key_value=self.api_key
            )
            fs = project.get_feature_store()
            return fs
        except Exception as e:
            logger.error(f"Failed to connect to Hopsworks: {str(e)}")
            raise CustomException(e, sys)

    def save_to_feature_group(self, df, group_name, version=6):
        """
        Saves the DataFrame to a Hopsworks Feature Group and maintains a local cache.
        
        If the existing feature group is a StreamFeatureGroup (requires Kafka),
        it is automatically recreated as a regular FeatureGroup (uses REST API)
        to avoid Kafka timeout errors from external clients.
        """
        try:
            # 1. Maintain a local cache in the data directory
            local_path = os.path.join("data", f"{group_name}.csv")
            os.makedirs("data", exist_ok=True)
            df_to_cache = df.copy()
            if 'ingestion_timestamp' in df_to_cache.columns:
                df_to_cache['ingestion_timestamp'] = pd.to_datetime(df_to_cache['ingestion_timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            if os.path.exists(local_path):
                try:
                    existing_df = pd.read_csv(local_path)
                    combined_df = pd.concat([existing_df, df_to_cache], ignore_index=True)
                    # Deduplicate on PKs to prevent double inserts
                    combined_df = combined_df.drop_duplicates(subset=["city", "ingestion_timestamp"])
                    combined_df.to_csv(local_path, index=False)
                    logger.info(f"Appended rows to local cache: {local_path}")
                except Exception as cache_err:
                    logger.warning(f"Failed to append to local cache: {cache_err}")
            else:
                df_to_cache.to_csv(local_path, index=False)
                logger.info(f"Initialized local cache: {local_path}")

            # 2. Push to Hopsworks
            fs = self.get_feature_store()
            
            # Get or create feature group
            aqi_fg = fs.get_or_create_feature_group(
                name=group_name,
                version=version,
                primary_key=["city", "ingestion_timestamp"],
                event_time="ingestion_timestamp",
                description="AQI and weather features for city"
            )
            
            # Safety check: StreamFeatureGroups require Kafka (port 9092) which
            # is blocked from external clients. If we got one, recreate as regular FG.
            fg_type = type(aqi_fg).__name__
            if fg_type == "StreamFeatureGroup":
                logger.warning(f"Feature group '{group_name}' v{version} is a StreamFeatureGroup. "
                              f"Recreating as regular FeatureGroup to avoid Kafka dependency...")
                aqi_fg.delete()
                aqi_fg = fs.create_feature_group(
                    name=group_name,
                    version=version,
                    primary_key=["city", "ingestion_timestamp"],
                    event_time="ingestion_timestamp",
                    description="AQI and weather features for city"
                )
                logger.info("Recreated as regular FeatureGroup successfully.")
            
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            # Insert data
            aqi_fg.insert(df)
            logger.info(f"Successfully inserted data into feature group: {group_name}")
            
        except Exception as e:
            logger.error(f"Failed to save data to feature store: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Example usage (requires API key)
    # from src.data_transformation.data_transformation import DataTransformation
    # from src.data_ingestion.data_ingestion import DataIngestion
    # ingestion = DataIngestion("shanghai")
    # transformer = DataTransformation()
    # df = transformer.transform(ingestion.get_city_data())
    # fs_ingestion = FeatureStoreIngestion()
    # fs_ingestion.save_to_feature_group(df, "aqi_features")
    pass
