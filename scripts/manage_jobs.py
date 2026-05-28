import os
import sys
sys.path.append(os.getcwd())
# pyrefly: ignore [missing-import]
import hopsworks
from dotenv import load_dotenv

def manage_jobs():
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

    print("Connecting to Hopsworks...")
    try:
        project = hopsworks.login(host=host, project=project_name, api_key_value=api_key)
        jobs_api = project.get_jobs_api()
    except Exception as e:
        print(f"Failed to connect to Hopsworks: {e}")
        sys.exit(1)

    print("\nFetching all jobs...")
    try:
        jobs = jobs_api.get_jobs()
    except Exception as e:
        print(f"Failed to fetch jobs: {e}")
        sys.exit(1)
    
    if not jobs:
        print("No jobs found in this project.")
        return

    print(f"\nFound {len(jobs)} jobs:")
    print("-" * 80)
    print(f"{'Job Name':<50} | {'Latest Run State':<20}")
    print("-" * 80)
    for job in jobs:
        try:
            executions = job.get_executions()
            state = executions[0].state if executions else "No executions"
        except Exception:
            state = "Unknown"
        print(f"{job.name:<50} | {state:<20}")
    print("-" * 80)

    # Identify stuck jobs
    stuck_jobs = []
    for job in jobs:
        try:
            executions = job.get_executions()
            if executions:
                latest = executions[0]
                # States that indicate the job is active/submitted/running
                if latest.state.upper() in ["SUBMITTED", "RUNNING", "ACCEPTED", "INITIALIZING"]:
                    stuck_jobs.append((job, latest))
        except Exception:
            pass

    if stuck_jobs:
        print("\n[WARNING] Detected currently active/stuck jobs:")
        for idx, (job, exec_info) in enumerate(stuck_jobs):
            print(f"[{idx}] Name: {job.name}")
            print(f"    Execution ID: {exec_info.id}")
            print(f"    State:        {exec_info.state}")
        
        print("\nWould you like to stop these active jobs to free up cluster resources?")
        print("You can run this script to list them, or use the Hopsworks UI to stop/kill them.")
        print("To stop a job programmatically, you can run:")
        print("  job.stop()  # inside Python")
        
        # Let's add a quick command line option to stop all of them
        choice = input("\nDo you want to stop ALL of the above active jobs now? (yes/no): ").strip().lower()
        if choice == 'yes':
            for job, exec_info in stuck_jobs:
                print(f"Stopping job '{job.name}' (Execution ID: {exec_info.id})...")
                try:
                    exec_info.stop()
                    print(f"Successfully sent stop request for '{job.name}'.")
                except Exception as e:
                    print(f"Failed to stop job '{job.name}': {e}")
        else:
            print("No jobs were stopped.")
    else:
        print("\n[SUCCESS] No active or stuck jobs detected.")

if __name__ == "__main__":
    manage_jobs()
