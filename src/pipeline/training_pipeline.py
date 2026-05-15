import os
import sys
sys.path.append(os.getcwd())
from src.data_ingestion.data_ingestion import DataIngestion
from src.data_transformation.data_transformation import DataTransformation
from src.data_ingestion.feature_store_ingestion import FeatureStoreIngestion
from src.model_training.model_trainer import ModelTrainer
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger("Training Pipeline")

class TrainingPipeline:
    def __init__(self, city_name="karachi"):
        self.city_name = city_name
        self.ingestion = DataIngestion(city_name)
        self.transformer = DataTransformation()
        self.fs_ingestion = FeatureStoreIngestion()
        self.trainer = ModelTrainer(city_name)

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
            try:
                self.fs_ingestion.save_to_feature_group(df, "aqi_features")
            except Exception as e:
                logger.error(f"Feature Store step failed: {str(e)}")
                # We can continue if we want to train on existing data, 
                # but usually we want fresh data.
                raise CustomException(e, sys)
            
            # Step 4: Model Training and Registration
            logger.info("Step 4: Training and registering model...")
            self.trainer.train_and_register()
            
            logger.info("Pipeline execution completed successfully")
            return df

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainingPipeline("karachi")
    pipeline.run_pipeline()
