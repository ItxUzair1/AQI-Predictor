import os
import hopsworks
from dotenv import load_dotenv

def delete_fg():
    load_dotenv()
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT_NAME")
    )
    fs = project.get_feature_store()
    try:
        fg = fs.get_feature_group("aqi_features", version=1)
        fg.delete()
        print("SUCCESS: Feature group 'aqi_features' deleted.")
    except Exception as e:
        print(f"INFO: Feature group not found or already deleted: {e}")

if __name__ == "__main__":
    delete_fg()
