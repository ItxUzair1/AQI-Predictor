import os
import sys
import logging
import pandas as pd
import hopsworks
from dotenv import load_dotenv

load_dotenv()

# Add the project root directory to the Python path
sys.path.append(os.getcwd())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def migrate_data():
    try:
        api_key = os.getenv("HOPSWORKS_API_KEY")
        project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "AQI_Prediction_System_10")
        host = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")

        if not api_key:
            print("ERROR: HOPSWORKS_API_KEY not set in .env")
            sys.exit(1)

        if sys.platform == "win32":
            local_tmp = os.path.join(os.getcwd(), "tmp")
            os.makedirs(local_tmp, exist_ok=True)
            os.environ["TMP"] = local_tmp
            os.environ["TEMP"] = local_tmp

        project = hopsworks.login(host=host, project=project_name, api_key_value=api_key)
        fs = project.get_feature_store()

        logger.info("Reading data from feature group v5...")
        try:
            v5_fg = fs.get_feature_group("aqi_features", version=5)
            # Read from offline store (might take a moment)
            df = v5_fg.read()
            logger.info(f"Read {len(df)} rows from v5.")
        except Exception as e:
            logger.error(f"Failed to read from v5: {e}")
            return

        # Ensure ingestion_timestamp exists
        if 'ingestion_timestamp' not in df.columns:
            logger.warning("'ingestion_timestamp' not found in v5. Using 'timestamp' as fallback.")
            df['ingestion_timestamp'] = df['timestamp']
            
        # Convert to datetime to ensure correct format
        df['ingestion_timestamp'] = pd.to_datetime(df['ingestion_timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # Drop duplicates on the new primary key to prevent insertion errors
        df = df.drop_duplicates(subset=["city", "ingestion_timestamp"])
        logger.info(f"After dropping duplicates on ['city', 'ingestion_timestamp'], {len(df)} rows remain.")

        logger.info("Getting or creating feature group v6...")
        v6_fg = fs.get_or_create_feature_group(
            name="aqi_features",
            version=6,
            primary_key=["city", "ingestion_timestamp"],
            event_time="ingestion_timestamp",
            description="AQI and weather features for city"
        )
        
        logger.info("Inserting data into v6...")
        v6_fg.insert(df, write_options={"wait_for_job": False})
        
        logger.info("Migration completed successfully!")
        print(f"SUCCESS: Migrated {len(df)} rows to v6.")
        print("Note: Spark materialization is running in the background on Hopsworks.")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    migrate_data()
