"""training_history.py

Simple persistent run-history logger for model training runs.

It appends one JSON object per line to `training_history.jsonl` in the project root.
Each record contains: timestamp, params, metrics, optional epoch history, and model paths.

This file is safe to import from notebooks or training scripts. Every training run should
call `append_run(...)` after completing training and evaluation so the run is recorded.
"""
from __future__ import annotations

import json
import os
import shutil
from tempfile import NamedTemporaryFile
from datetime import datetime
from typing import Any, Dict, Optional

HISTORY_PATH = "training_history.jsonl"

# Auto-export settings (disabled by default)
AUTO_EXPORT_N: Optional[int] = None
AUTO_EXPORT_CSV_PATH: str = "training_history.csv"


def append_run(params: Dict[str, Any], metrics: Dict[str, Any], history: Optional[Dict[str, Any]] = None, model_paths: Optional[Dict[str, str]] = None) -> None:
    """Append a training run to the persistent history file.

    params: hyperparameters and configuration used for this run
    metrics: final evaluation metrics (mae, mse, rmse, r2, accuracy, ...)
    history: (optional) training history dictionary with per-epoch lists
    model_paths: (optional) dict with saved model file paths (e.g., {'final': ..., 'best': ...})
    """
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "params": params,
        "metrics": metrics,
        "history": history,
        "model_paths": model_paths,
    }

    # Normalize JSON-serializable types (numpy floats/ints may appear)
    def _safe(o):
        try:
            json.dumps(o)
            return o
        except Exception:
            # Fallback to string for non-serializable objects
            return str(o)

    # Apply safe function recursively for top-level containers
    def _safe_map(obj):
        if isinstance(obj, dict):
            return {k: _safe_map(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_safe_map(v) for v in obj]
        return _safe(obj)

    safe_record = _safe_map(record)

    # Write the JSON line and ensure it's flushed to disk immediately to reduce
    # chance of partial writes or corruption (important if kernel/process dies).
    # Keep writes minimal and atomic-ish by flushing and fsync'ing.
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        line = json.dumps(safe_record, ensure_ascii=False) + "\n"
        f.write(line)
        try:
            f.flush()
            os.fsync(f.fileno())
        except Exception:
            # On some platforms (or environments) fsync may fail; ignore to
            # avoid breaking training flow but the flush above still helps.
            pass

    # If auto-export configured, export to CSV every AUTO_EXPORT_N runs
    try:
        global AUTO_EXPORT_N, AUTO_EXPORT_CSV_PATH
        if AUTO_EXPORT_N and isinstance(AUTO_EXPORT_N, int) and AUTO_EXPORT_N > 0:
            # Count runs and export if hit multiple
            total = len(load_history())
            if total % AUTO_EXPORT_N == 0:
                export_to_csv(AUTO_EXPORT_CSV_PATH)
    except Exception:
        # Do not raise during logging/export to avoid breaking training flow
        pass


def load_history() -> list:
    """Load all run records from the history file. Returns a list of dicts."""
    records = []
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    # Skip malformed lines but keep going
                    continue
    except FileNotFoundError:
        return []
    return records


def audit_history_file() -> Dict[str, int]:
    """Check the history file for malformed lines and return a brief report.

    Returns a dict with counts: total_lines, valid_json, malformed_lines.
    """
    total = 0
    valid = 0
    malformed = 0
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            for line in f:
                total += 1
                s = line.strip()
                if not s:
                    continue
                try:
                    json.loads(s)
                    valid += 1
                except Exception:
                    # try to salvage JSON payload if a prefix exists
                    idx = s.find('{')
                    if idx != -1:
                        try:
                            json.loads(s[idx:])
                            valid += 1
                            continue
                        except Exception:
                            pass
                    malformed += 1
    except FileNotFoundError:
        return {"total_lines": 0, "valid_json": 0, "malformed_lines": 0}
    return {"total_lines": total, "valid_json": valid, "malformed_lines": malformed}


def repair_history(backup: bool = True) -> int:
    """Attempt to repair `training_history.jsonl` by extracting valid JSON objects.

    If `backup` is True, the original file is copied to `training_history.jsonl.bak`.
    Returns the number of valid records written to the repaired file.
    """
    if not os.path.exists(HISTORY_PATH):
        return 0

    if backup:
        bak_path = HISTORY_PATH + ".bak"
        try:
            shutil.copy2(HISTORY_PATH, bak_path)
        except Exception:
            # ignore backup errors but continue
            pass

    valid_records = []
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # Try direct parse
            try:
                obj = json.loads(s)
                valid_records.append(obj)
                continue
            except Exception:
                pass

            # If line contains extra human-readable prefix, try to find first '{'
            idx = s.find('{')
            if idx != -1:
                try:
                    obj = json.loads(s[idx:])
                    valid_records.append(obj)
                    continue
                except Exception:
                    # If still fails, try to find further '{' occurrences (rare)
                    for i in range(idx + 1, len(s)):
                        if s[i] == '{':
                            try:
                                obj = json.loads(s[i:])
                                valid_records.append(obj)
                                break
                            except Exception:
                                continue
                    # otherwise skip this line
                    continue

    # Write repaired file atomically
    tmp = NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=os.path.dirname(HISTORY_PATH) or '.')
    try:
        with tmp as f:
            for r in valid_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp.name, HISTORY_PATH)
    finally:
        if os.path.exists(tmp.name):
            try:
                os.remove(tmp.name)
            except Exception:
                pass

    return len(valid_records)


def get_last_run() -> Optional[Dict[str, Any]]:
    """Return the last run record or None if no records."""
    records = load_history()
    if not records:
        return None
    return records[-1]


def print_recent_runs(n: int = 10) -> None:
    """Print a compact summary of the n most recent runs.

    Shows timestamp, a short params summary, MAE and RMSE (if present), and final model path.
    """
    records = load_history()
    if not records:
        print("No runs recorded yet (training_history.jsonl not found or empty).")
        return

    to_show = records[-n:]
    # Header
    header = f"Showing {len(to_show)} most recent runs (most recent last):"
    print(header)
    print("-" * len(header))
    for r in to_show:
        ts = r.get("timestamp")
        params = r.get("params") or {}
        # summarize params: pick up to 3 keys
        param_items = list(params.items())[:3]
        param_summary = ", ".join(f"{k}={v}" for k, v in param_items)
        metrics = r.get("metrics") or {}
        mae = metrics.get("mae")
        rmse = metrics.get("rmse")
        model_paths = r.get("model_paths") or {}
        final = model_paths.get("final") or model_paths.get("best") or "-"
        print(f"{ts} | {param_summary} | MAE={mae} | RMSE={rmse} | model={final}")


def export_to_csv(csv_path: str = "training_history.csv") -> None:
    """Export all runs to a CSV file with columns: timestamp, params(json), metrics(json), model_paths(json)."""
    import csv

    records = load_history()
    if not records:
        print("No records to export.")
        return

    with open(csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "params", "metrics", "model_paths", "history"])
        for r in records:
            writer.writerow([
                r.get("timestamp"),
                json.dumps(r.get("params"), ensure_ascii=False),
                json.dumps(r.get("metrics"), ensure_ascii=False),
                json.dumps(r.get("model_paths"), ensure_ascii=False),
                json.dumps(r.get("history"), ensure_ascii=False),
            ])

    print(f"Exported {len(records)} runs to {csv_path}")


def export_to_sqlite(db_path: str = "training_history.db") -> None:
    """Export runs into a simple SQLite database for querying.

    Table schema: runs(id INTEGER PRIMARY KEY, timestamp TEXT, params JSON, metrics JSON, model_paths JSON, history JSON)
    """
    import sqlite3

    records = load_history()
    if not records:
        print("No records to export.")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            params TEXT,
            metrics TEXT,
            model_paths TEXT,
            history TEXT
        )
        """
    )
    # Clear existing rows to avoid duplicates on repeated exports (optional behaviour)
    cur.execute("DELETE FROM runs")
    for r in records:
        cur.execute(
            "INSERT INTO runs (timestamp, params, metrics, model_paths, history) VALUES (?, ?, ?, ?, ?)",
            (
                r.get("timestamp"),
                json.dumps(r.get("params"), ensure_ascii=False),
                json.dumps(r.get("metrics"), ensure_ascii=False),
                json.dumps(r.get("model_paths"), ensure_ascii=False),
                json.dumps(r.get("history"), ensure_ascii=False),
            ),
        )

    conn.commit()
    conn.close()
    print(f"Exported {len(records)} runs into SQLite DB {db_path}")


def find_runs_by_param(param_name: str, value: Any) -> list:
    """Return runs where params[param_name] == value (simple equality match)."""
    records = load_history()
    matches = []
    for r in records:
        params = r.get("params") or {}
        if params.get(param_name) == value:
            matches.append(r)
    return matches


def enable_auto_export_csv(n: int, csv_path: Optional[str] = None) -> None:
    """Enable automatic export to CSV every n runs. If csv_path provided, use it.

    Example: enable_auto_export_csv(5, 'runs.csv') will export to runs.csv every 5 appended runs.
    """
    global AUTO_EXPORT_N, AUTO_EXPORT_CSV_PATH
    if n is None or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    AUTO_EXPORT_N = n
    if csv_path:
        AUTO_EXPORT_CSV_PATH = csv_path


def disable_auto_export_csv() -> None:
    """Disable automatic CSV export."""
    global AUTO_EXPORT_N
    AUTO_EXPORT_N = None


if __name__ == "__main__":
    print("training_history helper - append_run and load_history available")
