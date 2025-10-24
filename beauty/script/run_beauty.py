# run_beauty_at_16.py
import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

import requests  # make sure 'requests' is installed: pip install requests

# --- Pushover config ---
# Hardcoded Pushover credentials:
PUSHOVER_USER = "uy76bm9vr7w74cgz4hr69epfb9q9sb"
PUSHOVER_TOKEN = "aze67tfyszw84ozjtn1nc7bffzccg5"
PUSHOVER_API = "https://api.pushover.net/1/messages.json"

def notify(title: str, message: str, priority: int = 0):
    """Send a Pushover notification if creds exist; otherwise print to console."""
    if PUSHOVER_USER and PUSHOVER_TOKEN:
        try:
            resp = requests.post(
                PUSHOVER_API,
                data={
                    "user": PUSHOVER_USER,
                    "token": PUSHOVER_TOKEN,
                    "title": title,
                    "message": message,
                    "priority": priority,
                },
                timeout=10,
            )
            resp.raise_for_status()
        except Exception as e:
            print(f"[pushover] failed: {e}", file=sys.stderr)
    else:
        print(f"[notify] {title}\n{message}\n")

def wait_until(hour: int, minute: int):
    """Sleep until today at HH:MM local time. If already past, run immediately."""
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        # Already past today's target; run immediately
        print(f"[scheduler] It's already past {hour:02d}:{minute:02d}. Running now.")
        return
    delta = (target - now).total_seconds()
    print(f"[scheduler] Waiting until {target} to run ({int(delta)} seconds)...")
    time.sleep(delta)

def resolve_beauty_script() -> Path:
    """Find beauty_rolling_gathering.py near this file or by searching the project."""
    # 1) Same directory as this script
    here = Path(__file__).parent.resolve()
    candidate = here / "beauty_rolling_gathering.py"
    if candidate.exists():
        return candidate

    # 2) Try project root from current working directory
    cwd_candidate = Path.cwd() / "beauty_rolling_gathering.py"
    if cwd_candidate.exists():
        return cwd_candidate.resolve()

    # 3) Shallow search the repo (a couple of levels)
    for p in Path.cwd().rglob("beauty_rolling_gathering.py"):
        return p.resolve()

    for p in here.parent.rglob("beauty_rolling_gathering.py"):
        return p.resolve()

    raise FileNotFoundError("Could not locate beauty_rolling_gathering.py")

def run_beauty_daily(script_path: Path) -> subprocess.CompletedProcess:
    """Run the BEAUTY rolling gatherer in normal (non-backfill) mode via runpy, capturing output."""
    code = (
        "import runpy; "
        f"runpy.run_path(r'{script_path}', init_globals={{'DO_BACKFILL_TODAY': False}})"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(script_path.parent),
    )

def main():
    wait_until(16, 0)  # 4pm local time

    try:
        script_path = resolve_beauty_script()
    except Exception as e:
        notify("[BEAUTY] START FAILED", f"Could not find beauty_rolling_gathering.py\n{e}", priority=1)
        raise

    start_msg = (
        "Starting BEAUTY rolling gather (daily run)\n"
        f"Script: {script_path}\n"
        f"Runner: {sys.executable}\n"
        f"Start:  {datetime.now().isoformat(timespec='seconds')}"
    )
    notify("[BEAUTY] Gathering STARTED", start_msg)

    t0 = time.time()
    try:
        result = run_beauty_daily(script_path)
    except Exception as e:
        err = f"Runner crashed before completion:\n{e}"
        notify("[BEAUTY] Gathering CRASHED", err, priority=1)
        raise

    elapsed = time.time() - t0
    tail_stdout = (result.stdout or "").strip()
    tail_stderr = (result.stderr or "").strip()

    if result.returncode == 0:
        ok_msg = (
            "BEAUTY rolling gather FINISHED (OK)\n"
            f"Elapsed: {elapsed:.1f}s\n"
            f"Script:  {script_path}\n"
            "Last lines:\n"
            f"{tail_stdout[-1000:] or '(no stdout)'}"
        )
        notify("[BEAUTY] Gathering FINISHED", ok_msg)
        print(ok_msg)
    else:
        err_msg = (
            "BEAUTY rolling gather FAILED\n"
            f"Elapsed: {elapsed:.1f}s  Exit: {result.returncode}\n"
            f"Script:  {script_path}\n"
            "STDERR (tail):\n"
            f"{tail_stderr[-2000:] or '(no stderr)'}\n\n"
            "STDOUT (tail):\n"
            f"{tail_stdout[-1000:] or '(no stdout)'}"
        )
        notify("[BEAUTY] Gathering FAILED", err_msg, priority=1)
        print(err_msg, file=sys.stderr)
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()