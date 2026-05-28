import os
import sys
sys.path.append(os.getcwd())
# pyrefly: ignore [missing-import]
import hopsworks
from dotenv import load_dotenv


def delete_feature_group(version=4):
    """
    Deletes the aqi_features feature group from Hopsworks.
    This wipes ALL rows — use before re-backfilling with corrected data.
    """
    load_dotenv()

    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "AQI_Prediction_System_10")
    host = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")

    if not api_key:
        print("ERROR: HOPSWORKS_API_KEY not set in .env")
        sys.exit(1)

    # Windows workaround
    if sys.platform == "win32":
        local_tmp = os.path.join(os.getcwd(), "tmp")
        os.makedirs(local_tmp, exist_ok=True)
        os.environ["TMP"] = local_tmp
        os.environ["TEMP"] = local_tmp

    project = hopsworks.login(
        host=host,
        api_key_value=api_key,
        project=project_name
    )
    fs = project.get_feature_store()

    try:
        fg = fs.get_feature_group("aqi_features", version=version)
        fg.delete()
        print(f"SUCCESS: Feature group 'aqi_features' v{version} deleted (all old data wiped).")
    except Exception as e:
        print(f"INFO: Feature group v{version} not found or already deleted: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Delete aqi_features feature group from Hopsworks")
    parser.add_argument("--version", type=int, default=3, help="Feature group version to delete (default: 3)")
    args = parser.parse_args()
    delete_feature_group(version=args.version)
