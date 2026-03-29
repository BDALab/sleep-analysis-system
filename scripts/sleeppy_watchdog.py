import os
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(r"E:\geneactiv-processing-data")
DEBUG_LOG = ROOT / "debug.log"
LOG_DIR = ROOT / "media" / "watchdog"
WATCHDOG_LOG = LOG_DIR / "sleeppy_watchdog.log"
RUNNER_LOG = LOG_DIR / "sleeppy_runner.log"
PYTHON_EXE = ROOT / ".venv" / "Scripts" / "python.exe"
POLL_SECONDS = 600
STALE_AFTER = timedelta(minutes=15)
TIMESTAMP_RE = re.compile(r"^[A-Z]+\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")


def log(message):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    with WATCHDOG_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def tail_lines(path, max_bytes=512000):
    if not path.exists():
        return []
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - max_bytes))
        data = fh.read().decode("utf-8", errors="ignore")
    return data.splitlines()


def parse_timestamp(line):
    match = TIMESTAMP_RE.match(line)
    if not match:
        return None
    return datetime.strptime(
        f"{match.group(1)},{match.group(2)}",
        "%Y-%m-%d %H:%M:%S,%f",
    )


def latest_sleeppy_line():
    lines = tail_lines(DEBUG_LOG)
    for line in reversed(lines):
        if "sleeppy" in line.lower() or "SleepPy" in line:
            return parse_timestamp(line), line
    return None, None


def completed_after(start_time):
    for line in reversed(tail_lines(DEBUG_LOG)):
        if "SleepPy playground completed" in line:
            ts = parse_timestamp(line)
            if ts and ts >= start_time:
                return True
    return False


def spawn_runner():
    code = (
        "import os; "
        "os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings'); "
        "import django; django.setup(); "
        "from dashboard.logic.sleeppy.sleeppy import sleeppy_clean, sleeppy_all; "
        "sleeppy_clean(); "
        "sys_stdout = __import__('sys').stdout; "
        "sys_stdout.write('Starting sleeppy_all()\\n'); sys_stdout.flush(); "
        "result = sleeppy_all(); "
        "sys_stdout.write(f'sleeppy_all() finished: {result}\\n'); sys_stdout.flush()"
    )
    out = RUNNER_LOG.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        [str(PYTHON_EXE), "-c", code],
        cwd=str(ROOT),
        stdout=out,
        stderr=subprocess.STDOUT,
    )
    log(f"Started sleeppy runner pid={proc.pid}")
    return proc, out


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now()
    log("Watchdog started")
    runner = None
    runner_log_handle = None

    try:
        while True:
            if completed_after(start_time):
                log("Detected 'SleepPy playground completed'. Watchdog exiting.")
                break

            latest_ts, latest_line = latest_sleeppy_line()
            now = datetime.now()
            stale = latest_ts is None or now - latest_ts > STALE_AFTER

            if runner is not None and runner.poll() is not None:
                log(f"Runner exited with code {runner.returncode}")
                runner = None
                if runner_log_handle is not None:
                    runner_log_handle.close()
                    runner_log_handle = None

            if runner is None and stale:
                log(
                    "No recent Sleeppy activity detected. "
                    f"Latest line: {latest_line!r}. Restarting."
                )
                runner, runner_log_handle = spawn_runner()
            else:
                log(
                    "Watchdog check ok. "
                    f"runner_active={runner is not None}, latest_ts={latest_ts}, stale={stale}"
                )

            time.sleep(POLL_SECONDS)
    finally:
        if runner_log_handle is not None:
            runner_log_handle.close()


if __name__ == "__main__":
    main()
