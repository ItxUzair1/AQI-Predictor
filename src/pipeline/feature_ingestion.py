import os
import sys
sys.path.append(os.getcwd())
from src.data_ingestion.data_ingestion import DataIngestion
from src.data_transformation.data_transformation import DataTransformation
from src.data_ingestion.feature_store_ingestion import FeatureStoreIngestion
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger("Feature Ingestion Pipeline")

class FeatureIngestionPipeline:
    def __init__(self, city_name="karachi"):
        self.city_name = city_name
        self.ingestion = DataIngestion(city_name)
        self.transformer = DataTransformation()
        self.fs_ingestion = FeatureStoreIngestion()

    def run(self):
        try:
            logger.info(f"Starting feature ingestion for city: {self.city_name}")
            
            # Step 1: Ingestion
            logger.info("Step 1: Fetching raw data...")
            raw_data = self.ingestion.get_city_data()
            
            # Step 2: Transformation
            logger.info("Step 2: Transforming data...")
            df = self.transformer.transform(raw_data)
            
            # Step 3: Feature Store Storage
            logger.info("Step 3: Storing features...")
            self.fs_ingestion.save_to_feature_group(df, "aqi_features", version=5)
            
            logger.info("Feature ingestion completed successfully")
            return df

        except Exception as e:
            logger.error(f"Feature ingestion failed: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = FeatureIngestionPipeline("karachi")
    pipeline.run()
