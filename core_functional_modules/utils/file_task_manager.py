"""
File-level task manager for multi-GPU process isolation.

Multiple independent Python processes (launched via shell with CUDA_VISIBLE_DEVICES)
coordinate file-level work assignment through atomic directory-based locks:

  - Lock:   os.makedirs(lock_path)  — atomic on POSIX, fails if already exists
  - Done:   <output_path>.done      — marker file written after successful processing
  - Cleanup: cleanup_locks()        — remove stale .lock dirs from interrupted runs

Usage:
    from utils.file_task_manager import discover_tasks, process_all_tasks, cleanup_locks

    tasks = discover_tasks(input_dir, output_dir, suffix=".parquet", output_suffix=".jsonl")
    process_all_tasks(tasks, my_process_func)
"""

import os
import random
import shutil
from typing import Callable, Dict, List, Optional


def discover_tasks(
    input_dir: str,
    output_dir: str,
    suffix: str = ".parquet",
    output_suffix: str = ".jsonl",
) -> List[dict]:
    """
    Discover all input files and build a task list with lock/done paths.

    Each task dict contains:
        input_path  — absolute path to the input file
        output_path — corresponding output path under output_dir
        lock_path   — <output_path>.lock  (directory used as atomic lock)
        done_path   — <output_path>.done  (marker for completion)

    The output directory structure mirrors the input directory structure.
    """
    tasks: List[dict] = []

    if not os.path.isdir(input_dir):
        return tasks

    for root, _dirs, files in os.walk(input_dir):
        for fname in sorted(files):
            if not fname.endswith(suffix):
                continue

            input_path = os.path.join(root, fname)

            # Mirror directory structure
            rel_path = os.path.relpath(input_path, input_dir)
            base, _ = os.path.splitext(rel_path)
            output_path = os.path.join(output_dir, base + output_suffix)

            tasks.append({
                "input_path": input_path,
                "output_path": output_path,
                "lock_path": output_path + ".lock",
                "done_path": output_path + ".done",
            })

    return tasks


def claim_task(task: dict) -> bool:
    """
    Try to acquire a file-level lock for a task.

    Returns True if the lock was successfully acquired.
    Returns False if the task is already locked or already done.
    """
    # Already completed — skip
    if os.path.exists(task["done_path"]):
        return False

    # Try atomic lock via mkdir
    try:
        os.makedirs(task["lock_path"])
        return True
    except OSError:
        # Already locked by another process
        return False


def finish_task(task: dict) -> None:
    """Mark a task as completed: create .done marker, remove .lock directory."""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(task["output_path"]), exist_ok=True)

    # Write done marker
    with open(task["done_path"], "w") as f:
        f.write("done\n")

    # Remove lock directory
    if os.path.exists(task["lock_path"]):
        try:
            os.rmdir(task["lock_path"])
        except OSError:
            pass


def release_task(task: dict) -> None:
    """Release the lock without marking as done (for error recovery)."""
    if os.path.exists(task["lock_path"]):
        try:
            os.rmdir(task["lock_path"])
        except OSError:
            pass


def process_all_tasks(
    tasks: List[dict],
    process_func: Callable[[dict], None],
) -> dict:
    """
    Main processing loop: iterate through tasks, claim → process → finish.

    Tasks are shuffled each round to reduce contention between parallel processes.
    Each round scans all tasks; if any were claimed, starts another round.
    Exits immediately when no tasks could be claimed (all done or locked).

    Args:
        tasks:        List of task dicts from discover_tasks().
        process_func: Callable that takes a single task dict and processes it.
                      The function should read from task["input_path"] and
                      write results to task["output_path"].

    Returns:
        dict with keys: processed, skipped_done, skipped_locked, errors
    """
    stats = {"processed": 0, "skipped_done": 0, "skipped_locked": 0, "errors": 0}

    while True:
        claimed_this_round = 0

        shuffled = list(tasks)
        random.shuffle(shuffled)

        for task in shuffled:
            # Already done
            if os.path.exists(task["done_path"]):
                stats["skipped_done"] += 1
                continue

            if not claim_task(task):
                stats["skipped_locked"] += 1
                continue

            # Successfully claimed
            claimed_this_round += 1
            try:
                os.makedirs(os.path.dirname(task["output_path"]), exist_ok=True)
                process_func(task)
                finish_task(task)
                stats["processed"] += 1
            except Exception as e:
                print(f"[ERROR] Failed to process {task['input_path']}: {e}")
                import traceback
                traceback.print_exc()
                release_task(task)
                stats["errors"] += 1

        if claimed_this_round == 0:
            break  # Nothing to claim → all done or all locked → exit

    return stats


def cleanup_locks(output_dir: str) -> int:
    """
    Remove all .lock directories under output_dir.

    Call this before launching a new batch of workers to clean up
    stale locks from interrupted runs.

    Returns the number of lock directories removed.
    """
    cleaned = 0
    if not os.path.isdir(output_dir):
        return cleaned

    for root, dirs, _files in os.walk(output_dir):
        for d in dirs:
            if d.endswith(".lock"):
                lock_path = os.path.join(root, d)
                try:
                    os.rmdir(lock_path)
                    cleaned += 1
                except OSError:
                    # Non-empty or permission error — try harder
                    try:
                        shutil.rmtree(lock_path)
                        cleaned += 1
                    except OSError:
                        pass

    if cleaned:
        print(f"[file_task_manager] Cleaned {cleaned} stale lock(s) in {output_dir}")

    return cleaned


def get_task_status(tasks: List[dict]) -> dict:
    """
    Report task status counts.

    Returns:
        dict with keys: total, done, locked, pending
    """
    done = 0
    locked = 0
    pending = 0

    for t in tasks:
        if os.path.exists(t["done_path"]):
            done += 1
        elif os.path.isdir(t["lock_path"]):
            locked += 1
        else:
            pending += 1

    return {
        "total": len(tasks),
        "done": done,
        "locked": locked,
        "pending": pending,
    }
