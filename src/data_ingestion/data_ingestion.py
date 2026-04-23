import os
import sys
sys.path.append(os.getcwd())
import requests
from src.logger import get_logger
from src.exception import CustomException
from dotenv import load_dotenv

load_dotenv()
logger=get_logger("AQI Data Ingestion")


class DataIngestion():
    def __init__(self,city_name):
        self.city_name=city_name
        self.token=os.getenv("AQICN_TOKEN")

    def get_city_data(self):
        try:
            url=f"https://api.waqi.info/feed/{self.city_name}/?token={self.token}"
            response=requests.get(url)
            
            if response.status_code != 200:
                raise CustomException(f"HTTP Error: {response.status_code}", sys)

            data = response.json()
            
            if data.get("status") != "ok":
                error_info = data.get("data", "Unknown API error")
                logger.error(f"API Error for {self.city_name}: {error_info}")
                raise CustomException(f"API Error: {error_info}", sys)

            logger.info(f"Data ingestion for {self.city_name} completed successfully")
            return data

        except Exception as e:
            logger.error(f"Exception occurred during data ingestion: {str(e)}")
            raise CustomException(e, sys)


if __name__=="__main__":
    ingestion=DataIngestion("shanghai")
    data=ingestion.get_city_data()
    print(data)
