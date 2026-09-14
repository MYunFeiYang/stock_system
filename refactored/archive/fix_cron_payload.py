#!/usr/bin/env python3
"""
Fix cron job payloads: change stock system jobs from agentTurn to command.
- agentTurn → AI agent unreliable for shell commands (sandbox, exec failures)
- command → direct shell execution on Gateway host (deterministic, no LLM needed)
"""

import json
import sqlite3
import shutil
import os
from datetime import datetime

OPENCLAW_HOME = os.path.expanduser("~/.openclaw")
JOBS_FILE = os.path.join(OPENCLAW_HOME, "cron", "jobs.json.migrated")
DB_FILE = os.path.join(OPENCLAW_HOME, "state", "openclaw.sqlite")

STOCK_JOB_IDS = {
    "ca38d074-c454-4f80-adbe-f10b8c2fc3ed",  # 早盘分析
    "1d736a97-6ff1-4f1f-865a-e5882d61438c",  # 收盘复盘
    "cf7fa95f-64c0-4dec-b430-493bab61a45f",  # 定期清理
}

# Timeout mapping (seconds) - keep same as before
TIMEOUTS = {
    "morning": 1200,   # 20min
    "post_close": 2400,  # 40min
    "cleanup": 180,    # 3min
}

def backup_file(path):
    backup = path + ".backup." + datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(path, backup)
    print(f"  Backup: {backup}")

def fix_jobs_json():
    """Update jobs.json.migrated — change agentTurn to command."""
    print("--- Fixing jobs.json.migrated ---")
    backup_file(JOBS_FILE)

    with open(JOBS_FILE, "r") as f:
        data = json.load(f)

    for job in data["jobs"]:
        if job["id"] not in STOCK_JOB_IDS:
            continue

        payload = job.get("payload", {})
        if payload.get("kind") != "agentTurn":
            print(f"  {job['name']}: already {payload.get('kind')}, skipping")
            continue

        message = payload.get("message", "")
        old_timeout = payload.get("timeoutSeconds", 600)
        command = _extract_command(message)

        # Build new command payload
        new_payload = {
            "kind": "command",
            "command": command,
            "timeoutSeconds": old_timeout,
        }

        job["payload"] = new_payload
        print(f"  {job['name']}: agentTurn → command")
        print(f"    command: {command[:80]}...")
        print(f"    timeout: {old_timeout}s")

    with open(JOBS_FILE, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("  ✓ jobs.json.migrated updated\n")

def _extract_command(message: str) -> str:
    """Extract shell command from agentTurn message."""
    if "：" in message:
        return message.split("：", 1)[1].strip()
    elif ": " in message:
        return message.split(": ", 1)[1].strip()
    return message.strip()


def fix_sqlite():
    """
    Update SQLite cron_jobs:
    - payload_kind: 'agentTurn' → 'command'
    - payload_message: strip prefix, keep only shell command
    - job_json: update payload in the full JSON blob
    """
    print("--- Fixing SQLite cron_jobs ---")
    backup_file(DB_FILE)

    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA busy_timeout = 5000")

    try:
        # Get all stock jobs
        placeholders = ",".join("?" for _ in STOCK_JOB_IDS)
        rows = conn.execute(
            f"SELECT job_id, name, payload_kind, payload_message, "
            f"payload_timeout_seconds, job_json "
            f"FROM cron_jobs WHERE job_id IN ({placeholders})",
            list(STOCK_JOB_IDS)
        ).fetchall()

        print(f"  Found {len(rows)} stock cron jobs")

        updated = 0
        for job_id, name, kind, message, timeout, job_json_str in rows:
            if kind != "agentTurn":
                print(f"  {name}: already kind={kind}, skipping")
                continue

            command = _extract_command(message or "")

            # Update flat columns
            conn.execute(
                "UPDATE cron_jobs SET payload_kind = ?, payload_message = ? "
                "WHERE job_id = ?",
                ("command", command, job_id)
            )

            # Update job_json if present
            if job_json_str:
                try:
                    job = json.loads(job_json_str)
                    payload = job.get("payload", {})
                    if payload.get("kind") == "agentTurn":
                        payload["kind"] = "command"
                        payload["command"] = command
                        payload.pop("message", None)
                        payload.pop("model", None)
                        payload.pop("thinking", None)
                        job["payload"] = payload
                        conn.execute(
                            "UPDATE cron_jobs SET job_json = ? WHERE job_id = ?",
                            (json.dumps(job, ensure_ascii=False), job_id)
                        )
                except (json.JSONDecodeError, KeyError):
                    pass

            updated += 1
            print(f"  {name}: agentTurn → command")
            print(f"    command: {command[:80]}...")

        if updated == 0:
            print("  WARN: No SQLite rows updated")
        else:
            conn.commit()
            print(f"  ✓ {updated} SQLite rows updated")

        # Verify
        for job_id, name, kind in conn.execute(
            f"SELECT job_id, name, payload_kind FROM cron_jobs "
            f"WHERE job_id IN ({placeholders})",
            list(STOCK_JOB_IDS)
        ):
            print(f"  Verify: {name} → kind={kind}")

    finally:
        conn.close()

if __name__ == "__main__":
    print("=" * 60)
    print("Fixing stock cron jobs: agentTurn → command")
    print("=" * 60)
    fix_jobs_json()
    fix_sqlite()
    print("\nDone! Restart OpenClaw Gateway for changes to take effect.")
