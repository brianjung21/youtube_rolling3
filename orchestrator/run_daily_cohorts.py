# run_beauty_then_fnb_then_kpop.py
import sys
import time
import subprocess
import threading
from types import SimpleNamespace
from pathlib import Path
from datetime import datetime, timedelta
import os
import signal
from zoneinfo import ZoneInfo
import fcntl

def _fmt_td(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

import requests  # pip install requests

# --- Absolute script paths you provided ---
BEAUTY_SCRIPT_PATH = Path("/Users/sanghoonjung/PycharmProjects/PythonProject/youtube_rolling3/beauty/script/beauty_rolling_gathering.py")
FNB_SCRIPT_PATH = Path("/Users/sanghoonjung/PycharmProjects/PythonProject/youtube_rolling3/fnb/script/fnb_rolling_gathering.py")
KPOP_SCRIPT_PATH   = Path("/Users/sanghoonjung/PycharmProjects/PythonProject/youtube_rolling3/kpop/script/kpop_rolling_gathering.py")

# --- Pushover config (hardcoded, as requested) ---
PUSHOVER_USER = "uy76bm9vr7w74cgz4hr69epfb9q9sb"
PUSHOVER_TOKEN = "aze67tfyszw84ozjtn1nc7bffzccg5"
PUSHOVER_API = "https://api.pushover.net/1/messages.json"

# --- Timezone & single-instance lock ---
KST = ZoneInfo("Asia/Seoul")  # Korea does not observe DST, but we pin the zone explicitly.
LOCKFILE_PATH = "/tmp/run_daily_cohorts.lock"

def notify(title: str, message: str, priority: int = 0):
    """Send a Pushover notification (hardcoded creds)."""
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


# --- Single-instance lock and signal handlers ---
def _acquire_single_instance_lock():
    """Prevent multiple concurrent schedulers.
    Creates/locks a file at LOCKFILE_PATH for the lifetime of this process.
    """
    lf = open(LOCKFILE_PATH, "w")
    try:
        fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"[lock] Another instance appears to be running (lock: {LOCKFILE_PATH}). Exiting.")
        sys.exit(1)

    # Write PID for observability
    lf.truncate(0)
    lf.write(str(os.getpid()))
    lf.flush()

    # Keep the file handle open so the lock persists for the process lifetime
    return lf

def _install_signal_handlers():
    def _handler(signum, frame):
        print(f"[signal] Received {signum}. Shutting down gracefully…")
        try:
            notify("[SCHEDULER] STOPPING", f"Signal {signum} received at {datetime.now(tz=KST).isoformat(timespec='seconds')}")
        except Exception:
            pass
        # Lock is released automatically when process exits; just exit now.
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass

def wait_until(hour: int, minute: int, tolerance_seconds: int = 15):
    """Sleep until the next occurrence of HH:MM in Asia/Seoul time.
    If current time is within `tolerance_seconds` after the target, proceed immediately.
    Prints periodic countdowns (hourly; every minute for the last 5 minutes).
    """
    while True:
        now_kst = datetime.now(tz=KST)
        target_today_kst = now_kst.replace(hour=hour, minute=minute, second=0, microsecond=0)

        if now_kst >= target_today_kst:
            lag = (now_kst - target_today_kst).total_seconds()
            if lag <= tolerance_seconds:
                print(f"[scheduler] Target {target_today_kst.isoformat()} reached in KST (lag {lag:.2f}s ≤ tolerance). Proceeding now.")
                return
            # schedule tomorrow's target in KST
            target_kst = (now_kst + timedelta(days=1)).replace(hour=hour, minute=minute, second=0, microsecond=0)
            print(f"[scheduler] Missed today's {hour:02d}:{minute:02d} KST by {lag:.2f}s; rolling to next day at {target_kst.isoformat()}.")
        else:
            target_kst = target_today_kst

        # Compute remaining seconds in real time using monotonic clock for accuracy over sleep
        def _remaining_secs():
            return max(0, int((target_kst - datetime.now(tz=KST)).total_seconds()))

        remaining = _remaining_secs()
        print(f"[scheduler] Next run target (KST): {target_kst.isoformat()} ({_fmt_td(remaining)} remaining)")

        # Sleep in chunks; print updates hourly and during final 5 minutes
        while remaining > 0:
            step = 60 if remaining > 60 else remaining
            time.sleep(step)
            remaining = _remaining_secs()
            if remaining > 0 and (remaining % 3600 == 0 or remaining <= 300):
                print(f"[scheduler] Time remaining until {target_kst.isoformat()}: {_fmt_td(remaining)}")

        return

def wait_until_next_day(hour: int, minute: int):
    """Sleep until tomorrow's HH:MM in Asia/Seoul time; print periodic countdowns."""
    now_kst = datetime.now(tz=KST)
    target_kst = (now_kst + timedelta(days=1)).replace(hour=hour, minute=minute, second=0, microsecond=0)

    def _remaining_secs():
        return max(0, int((target_kst - datetime.now(tz=KST)).total_seconds()))

    remaining = _remaining_secs()
    print(f"[scheduler] Next-day target (KST): {target_kst.isoformat()} ({_fmt_td(remaining)} remaining)")
    while remaining > 0:
        step = 60 if remaining > 60 else remaining
        time.sleep(step)
        remaining = _remaining_secs()
        if remaining > 0 and (remaining % 3600 == 0 or remaining <= 300):
            print(f"[scheduler] Time remaining until {target_kst.isoformat()}: {_fmt_td(remaining)}")


def start_command_listener():
    """Spawn a daemon thread that listens to stdin and responds to certain probes.
    Typing exactly 'still running?' + Enter will print a status line.
    """
    def _listener():
        try:
            for line in sys.stdin:
                if line.strip() == "still running?":
                    print("yes, still up and waiting... damn it.")
                else:
                    print("Don't bother me...")
        except Exception as e:
            # Do not crash the main loop if stdin is closed/not interactive
            print(f"[listener] stopped: {e}")
    t = threading.Thread(target=_listener, name="stdin-listener", daemon=True)
    t.start()

def run_gatherer(script_path: Path, tag: str):
    """Run a gatherer by invoking the script directly, streaming output live and capturing it.
    Returns an object with .returncode, .stdout, .stderr similar to subprocess.CompletedProcess.
    """
    proc = subprocess.Popen(
        [sys.executable, "-u", str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=str(script_path.parent),
    )

    stdout_buf = []
    stderr_buf = []

    def _pump(stream, buf, prefix):
        for line in iter(stream.readline, ""):
            buf.append(line)
            try:
                print(line, end="")  # stream to parent console
            except Exception:
                # Best-effort; ignore printing errors
                pass
        stream.close()

    t_out = threading.Thread(target=_pump, args=(proc.stdout, stdout_buf, "STDOUT"), daemon=True)
    t_err = threading.Thread(target=_pump, args=(proc.stderr, stderr_buf, "STDERR"), daemon=True)
    t_out.start(); t_err.start()
    proc.wait()
    t_out.join(); t_err.join()

    return SimpleNamespace(
        returncode=proc.returncode,
        stdout="".join(stdout_buf),
        stderr="".join(stderr_buf),
    )

def run_one(script_path: Path, tag: str) -> None:
    """Run one sector with notifications and error handling."""
    if not script_path.exists():
        notify(f"[{tag}] START FAILED", f"Not found: {script_path}", priority=1)
        raise FileNotFoundError(f"{script_path}")

    start_msg = (
        f"Starting {tag} rolling gather (daily run)\n"
        f"Script: {script_path}\n"
        f"Runner: {sys.executable}\n"
        f"Start (KST):  {datetime.now(tz=KST).isoformat(timespec='seconds')}"
    )
    notify(f"[{tag}] Gathering STARTED", start_msg)
    t0 = time.time()

    try:
        res = run_gatherer(script_path, tag)
    except Exception as e:
        notify(f"[{tag}] Gathering CRASHED", f"Runner crashed before completion:\n{e}", priority=1)
        raise

    elapsed = time.time() - t0
    tail_stdout = (res.stdout or "").strip()
    tail_stderr = (res.stderr or "").strip()

    if res.returncode == 0:
        ok_msg = (
            f"{tag} rolling gather FINISHED (OK)\n"
            f"Elapsed: {elapsed:.1f}s\n"
            f"Script:  {script_path}\n"
            "Last lines:\n"
            f"{tail_stdout[-1000:] or '(no stdout)'}"
        )
        notify(f"[{tag}] Gathering FINISHED", ok_msg)
        print(ok_msg)
    else:
        err_msg = (
            f"{tag} rolling gather FAILED\n"
            f"Elapsed: {elapsed:.1f}s  Exit: {res.returncode}\n"
            f"Script:  {script_path}\n"
            "STDERR (tail):\n"
            f"{tail_stderr[-2000:] or '(no stderr)'}\n\n"
            "STDOUT (tail):\n"
            f"{tail_stdout[-1000:] or '(no stdout)'}"
        )
        notify(f"[{tag}] Gathering FAILED", err_msg, priority=1)
        print(err_msg, file=sys.stderr)
        sys.exit(res.returncode)

def main():
    # Enforce single instance and graceful shutdown
    _install_signal_handlers()
    _lock_handle = _acquire_single_instance_lock()

    # Start background stdin listener for 'still running?' probes
    start_command_listener()
    while True:
        # Wait until 4pm local time (today)
        wait_until(16, 0)
        print(f"[scheduler] Woke at (KST): {datetime.now(tz=KST).isoformat(timespec='seconds')}")
        print("[scheduler] Starting daily sector runs: BEAUTY → FNB → KPOP")

        # 1) BEAUTY
        run_one(BEAUTY_SCRIPT_PATH, "BEAUTY")

        # 2) FNB
        run_one(FNB_SCRIPT_PATH, "FNB")

        # 3) KPOP
        run_one(KPOP_SCRIPT_PATH, "KPOP")

        # Final daily summary (only after all three finish successfully)
        notify(
            "[DAILY] Gathering COMPLETED",
            (
                "All daily gatherers finished successfully.\n"
                f"Beauty script: {BEAUTY_SCRIPT_PATH}\n"
                f"FNB script:    {FNB_SCRIPT_PATH}\n"
                f"KPOP script:   {KPOP_SCRIPT_PATH}\n"
                f"End (KST):      {datetime.now(tz=KST).isoformat(timespec='seconds')}"
            )
        )

        # Sleep until next day's 4pm before looping again
        wait_until_next_day(16, 0)

if __name__ == "__main__":
    main()