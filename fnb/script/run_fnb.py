# run_fnb_at_16.py
import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

import requests  # make sure 'requests' is installed: pip install requests

KPOP_SCRIPT_PATH = Path("/Users/sanghoonjung/PycharmProjects/PythonProject/youtube_rolling3/kpop/script/kpop_rolling_gathering.py")

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

def resolve_fnb_script() -> Path:
    """Find fnb_rolling_gathering.py near this file or by searching the project."""
    # 1) Same directory as this script
    here = Path(__file__).parent.resolve()
    candidate = here / "fnb_rolling_gathering.py"
    if candidate.exists():
        return candidate

    # 2) Try project root from current working directory
    cwd_candidate = Path.cwd() / "fnb_rolling_gathering.py"
    if cwd_candidate.exists():
        return cwd_candidate.resolve()

    # 3) Shallow search the repo (a couple of levels)
    for p in Path.cwd().rglob("fnb_rolling_gathering.py"):
        return p.resolve()

    for p in here.parent.rglob("fnb_rolling_gathering.py"):
        return p.resolve()

    raise FileNotFoundError("Could not locate fnb_rolling_gathering.py")

def run_fnb_daily(script_path: Path) -> subprocess.CompletedProcess:
    """Run the FNB gatherer in normal (non-backfill) mode via runpy, capturing output."""
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

def run_kpop_daily(script_path: Path) -> subprocess.CompletedProcess:
    """Run the KPOP gatherer in normal (non-backfill) mode via runpy, capturing output."""
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
        script_path = resolve_fnb_script()
    except Exception as e:
        notify("[FNB] START FAILED", f"Could not find fnb_rolling_gathering.py\n{e}", priority=1)
        raise

    start_msg = (
        "Starting FNB rolling gather (daily run)\n"
        f"Script: {script_path}\n"
        f"Runner: {sys.executable}\n"
        f"Start:  {datetime.now().isoformat(timespec='seconds')}"
    )
    notify("[FNB] Gathering STARTED", start_msg)

    t0 = time.time()
    try:
        result = run_fnb_daily(script_path)
    except Exception as e:
        err = f"Runner crashed before completion:\n{e}"
        notify("[FNB] Gathering CRASHED", err, priority=1)
        raise

    elapsed = time.time() - t0
    tail_stdout = (result.stdout or "").strip()
    tail_stderr = (result.stderr or "").strip()

    if result.returncode == 0:
        ok_msg = (
            "FNB rolling gather FINISHED (OK)\n"
            f"Elapsed: {elapsed:.1f}s\n"
            f"Script:  {script_path}\n"
            "Last lines:\n"
            f"{tail_stdout[-1000:] or '(no stdout)'}"
        )
        notify("[FNB] Gathering FINISHED", ok_msg)
        print(ok_msg)

        # --- Run KPOP gatherer next ---
        try:
            kpop_path = KPOP_SCRIPT_PATH
            if not kpop_path.exists():
                raise FileNotFoundError(f"Not found: {kpop_path}")
        except Exception as e:
            notify("[KPOP] START FAILED", f"Could not find kpop_rolling_gathering.py\n{e}", priority=1)
            raise

        k_start = (
            "Starting KPOP rolling gather (daily run)\n"
            f"Script: {kpop_path}\n"
            f"Runner: {sys.executable}\n"
            f"Start:  {datetime.now().isoformat(timespec='seconds')}"
        )
        notify("[KPOP] Gathering STARTED", k_start)

        t1 = time.time()
        try:
            k_res = run_kpop_daily(kpop_path)
        except Exception as e:
            k_err = f"Runner crashed before completion:\n{e}"
            notify("[KPOP] Gathering CRASHED", k_err, priority=1)
            raise

        k_elapsed = time.time() - t1
        k_stdout = (k_res.stdout or "").strip()
        k_stderr = (k_res.stderr or "").strip()

        if k_res.returncode == 0:
            k_ok = (
                "KPOP rolling gather FINISHED (OK)\n"
                f"Elapsed: {k_elapsed:.1f}s\n"
                f"Script:  {kpop_path}\n"
                "Last lines:\n"
                f"{k_stdout[-1000:] or '(no stdout)'}"
            )
            notify("[KPOP] Gathering FINISHED", k_ok)
            print(k_ok)
        else:
            k_fail = (
                "KPOP rolling gather FAILED\n"
                f"Elapsed: {k_elapsed:.1f}s  Exit: {k_res.returncode}\n"
                f"Script:  {kpop_path}\n"
                "STDERR (tail):\n"
                f"{k_stderr[-2000:] or '(no stderr)'}\n\n"
                "STDOUT (tail):\n"
                f"{k_stdout[-1000:] or '(no stdout)'}"
            )
            notify("[KPOP] Gathering FAILED", k_fail, priority=1)
            print(k_fail, file=sys.stderr)
            sys.exit(k_res.returncode)

        # --- Final daily completion message ---
        notify(
            "[DAILY] Gathering COMPLETED",
            (
                "All daily gatherers finished successfully.\n"
                f"FNB script:  {script_path}\n"
                f"KPOP script: {kpop_path}\n"
                f"End:        {datetime.now().isoformat(timespec='seconds')}"
            )
        )
    else:
        err_msg = (
            "FNB rolling gather FAILED\n"
            f"Elapsed: {elapsed:.1f}s  Exit: {result.returncode}\n"
            f"Script:  {script_path}\n"
            "STDERR (tail):\n"
            f"{tail_stderr[-2000:] or '(no stderr)'}\n\n"
            "STDOUT (tail):\n"
            f"{tail_stdout[-1000:] or '(no stdout)'}"
        )
        notify("[FNB] Gathering FAILED", err_msg, priority=1)
        print(err_msg, file=sys.stderr)
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()