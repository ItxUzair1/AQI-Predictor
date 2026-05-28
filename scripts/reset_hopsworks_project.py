import os
import sys
sys.path.append(os.getcwd())
# pyrefly: ignore [missing-import]
import hopsworks
from dotenv import load_dotenv

def reset_project():
    load_dotenv()

    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "AQI_Prediction_System_10")
    host = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")

    if not api_key:
        print("[ERROR] HOPSWORKS_API_KEY not found in env")
        sys.exit(1)

    # Windows workaround
    if sys.platform == "win32":
        local_tmp = os.path.join(os.getcwd(), "tmp")
        os.makedirs(local_tmp, exist_ok=True)
        os.environ["TMP"] = local_tmp
        os.environ["TEMP"] = local_tmp

    print(f"Connecting to Hopsworks project: {project_name}...")
    try:
        project = hopsworks.login(host=host, project=project_name, api_key_value=api_key)
        fs = project.get_feature_store()
        jobs_api = project.get_jobs_api()
    except Exception as e:
        print(f"[ERROR] Failed to connect: {e}")
        sys.exit(1)

    # Step 1: Stop all active/submitted executions to free up cluster resources
    print("\n[STEP 1] Fetching all jobs to stop active executions...")
    try:
        jobs = jobs_api.get_jobs()
        if jobs:
            for job in jobs:
                try:
                    executions = job.get_executions()
                    for execution in executions:
                        if execution.state.upper() in ["SUBMITTED", "RUNNING", "ACCEPTED", "INITIALIZING"]:
                            print(f"Stopping active execution {execution.id} for job: {job.name}...")
                            execution.stop()
                            print(f"-> Stop command sent successfully.")
                except Exception as ex:
                    print(f"Could not check/stop executions for job {job.name}: {ex}")
        else:
            print("No jobs found in the project.")
    except Exception as e:
        print(f"Failed to process jobs: {e}")

    # Step 2: Delete all versions of the feature group to wipe data
    print("\n[STEP 2] Deleting all versions of feature group 'aqi_features'...")
    for version in [1, 2, 3, 4]:
        try:
            fg = fs.get_feature_group("aqi_features", version=version)
            print(f"Deleting feature group 'aqi_features' version {version}...")
            fg.delete()
            print(f"-> Version {version} deleted successfully.")
        except Exception:
            # Silence if group version doesn't exist
            pass

    print("\n[SUCCESS] Hopsworks project reset complete! All jobs stopped and feature groups deleted.")
    print("You can now run backfill fresh.")

if __name__ == "__main__":
    reset_project()
