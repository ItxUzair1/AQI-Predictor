import os
import sys
sys.path.append(os.getcwd())
# pyrefly: ignore [missing-import]
import hopsworks
from dotenv import load_dotenv

def main():
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
        try:
            if not os.path.exists("/tmp"):
                os.makedirs("/tmp", exist_ok=True)
        except:
            pass

    print("Connecting to Hopsworks...")
    try:
        project = hopsworks.login(host=host, project=project_name, api_key_value=api_key)
        fs = project.get_feature_store()
        print(f"Connected successfully to project: {project_name}")
    except Exception as e:
        print(f"Failed to connect to Hopsworks: {e}")
        sys.exit(1)

    print("\nFetching feature groups...")
    try:
        # Change version to 4 since that's what we are now using
        aqi_fg = fs.get_feature_group("aqi_features", version=5)
        fgs = fs.get_feature_groups()
        if not fgs:
            print("No feature groups found in the project.")
            return
        
        print(f"Found {len(fgs)} feature group(s):")
        print("=" * 80)
        
        for fg in fgs:
            print(f"Feature Group: {fg.name} (v{fg.version})")
            print(f"  Description: {fg.description}")
            print(f"  Primary Keys: {fg.primary_key}")
            print(f"  Event Time: {fg.event_time}")
            
            # Try to read some metadata / rows
            try:
                print("  Reading data...")
                df = fg.select_all().read()
                print(f"  -> Rows: {len(df)}, Columns: {len(df.columns)}")
                if len(df) > 0:
                    print("  -> Date range:")
                    if fg.event_time in df.columns:
                        import pandas as pd
                        times = pd.to_datetime(df[fg.event_time])
                        print(f"     Start: {times.min()}")
                        print(f"     End:   {times.max()}")
                    else:
                        print("     Event time column not found in data.")
                    print("  -> First 3 rows sample:")
                    # print nicely
                    print(df.head(3).to_string(index=False))
                else:
                    print("  -> Feature Group is empty.")
            except Exception as e:
                print(f"  -> Failed to read data: {e}")
            print("-" * 80)
            
    except Exception as e:
        print(f"Error querying feature groups: {e}")

if __name__ == "__main__":
    main()
