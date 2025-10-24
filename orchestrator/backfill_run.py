# daily_runner.py
import subprocess
import time
from datetime import datetime, timedelta, timezone

# KST timezone (UTC+9)
KST = timezone(timedelta(hours=9))

# Paths to your scripts
SCRIPTS = [
    "/Users/sanghoonjung/PycharmProjects/PythonProject/youtube_rolling3/kpop/script/kpop_rolling_gathering.py",
    "/Users/sanghoonjung/PycharmProjects/PythonProject/youtube_rolling3/beauty/script/beauty_rolling_gathering.py",
    "/Users/sanghoonjung/PycharmProjects/PythonProject/youtube_rolling3/fnb/script/fnb_rolling_gathering.py",
]

def wait_until_4pm():
    """Sleep until the next 4pm KST."""
    now = datetime.now(KST)
    target = now.replace(hour=16, minute=0, second=0, microsecond=0)
    if now >= target:  # already past 4pm → next day
        target += timedelta(days=1)
    wait_seconds = (target - now).total_seconds()
    print(f"[scheduler] Waiting {wait_seconds/3600:.2f} hours until {target.strftime('%Y-%m-%d %H:%M')} KST...")
    time.sleep(wait_seconds)

def run_scripts():
    """Run all specified scripts sequentially."""
    for path in SCRIPTS:
        print(f"[scheduler] Running {path} ...")
        try:
            subprocess.run(["/opt/anaconda3/envs/python391/bin/python", path], check=True)
            print(f"[scheduler] Finished {path}")
        except subprocess.CalledProcessError as e:
            print(f"[scheduler] ERROR running {path}: {e}")

def main():
    print("[scheduler] Starting daily runner (waiting for 4pm KST)")
    while True:
        wait_until_4pm()
        run_scripts()
        print("[scheduler] All scripts done. Waiting for next 4pm cycle.\n")
        time.sleep(60)  # prevent immediate loop rerun

if __name__ == "__main__":
    main()