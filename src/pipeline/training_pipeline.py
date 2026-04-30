import os
import sys
sys.path.append(os.getcwd())
from src.data_ingestion.data_ingestion import DataIngestion
from src.data_transformation.data_transformation import DataTransformation
from src.data_ingestion.feature_store_ingestion import FeatureStoreIngestion
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger("Training Pipeline")

class TrainingPipeline:
    def __init__(self, city_name="shanghai"):
        self.city_name = city_name
        self.ingestion = DataIngestion(city_name)
        self.transformer = DataTransformation()
        self.fs_ingestion = FeatureStoreIngestion()

    def run_pipeline(self):
        try:
            logger.info(f"Starting pipeline for city: {self.city_name}")
            
            # Step 1: Ingestion
            logger.info("Step 1: Fetching raw data...")
            raw_data = self.ingestion.get_city_data()
            
            # Step 2: Transformation
            logger.info("Step 2: Transforming data and computing features...")
            df = self.transformer.transform(raw_data)
            
            # Step 3: Feature Store Storage
            logger.info("Step 3: Storing features in Feature Store...")
            # Note: This will fail if hopsworks is not installed or API key is missing
            try:
                self.fs_ingestion.save_to_feature_group(df, "aqi_features")
            except Exception as e:
                logger.warning(f"Feature Store step failed (likely missing hopsworks or API key): {str(e)}")
                print("\n[WARNING] Feature Store ingestion failed. Please ensure 'hopsworks' is installed and API keys are set.")
                print("\nComputed Features:")
                try:
                    print(df.to_string())
                except UnicodeEncodeError:
                    # Fallback for terminals that can't handle non-ASCII characters (e.g. Chinese characters in city names)
                    print(df.to_string().encode('ascii', 'replace').decode('ascii'))
            
            logger.info("Pipeline execution completed")
            return df

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainingPipeline("karachi")
    pipeline.run_pipeline()
