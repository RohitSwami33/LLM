#!/usr/bin/env python3
"""Wait until the OpenRouter free-tier window resets, then resume generation.

Auto-resumes the `corpus` run (state is saved per batch; completed IDs are skipped).
Usage: python3 resume_watcher.py [target_time]
"""
import os
import subprocess
import sys
import time
from datetime import datetime, timezone


def main():
    target = float(sys.argv[1]) if len(sys.argv) > 1 else 1786924800000 / 1000.0
    now = time.time()
    wait = max(0, target - now)
    print(f"[{datetime.now()}] waiting {wait/60:.1f} min for rate-limit reset", flush=True)
    while time.time() < target:
        time.sleep(60)
    print(f"[{datetime.now()}] reset reached; resuming corpus generation", flush=True)
    env = os.environ.copy()
    env.update({
        "LLM_API_KEY_FILE": os.path.expanduser("~/.openrouter_api_key"),
        "DISTILL_FORCE_TEACHER": "dots-studio/dots-3-note-preview:free",
        "DISTILL_CONCURRENCY": "2",   # stay under the 50/window limit
        "DISTILL_TIMEOUT": "180",
        "DISTILL_MAX_TOKENS": "3000",
        "DISTILL_BATCH": "4",
    })
    r = subprocess.run(
        [sys.executable, "-u", "generate.py", "240", "corpus"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=env,
    )
    print(f"[{datetime.now()}] corpus run finished rc={r.returncode}", flush=True)
    # loop: keep resuming with the window cadence
    main()


if __name__ == "__main__":
    main()