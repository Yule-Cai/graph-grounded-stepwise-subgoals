#!/usr/bin/env python3
"""Live progress dashboard for Paper A preference-sweep experiments.

This monitor is read-only: it inspects episode CSVs and tee logs produced by
``run_preference_250_upgrade.zsh`` / ``run_language_cost_option_baselines.zsh``.
It is intended to run in a second terminal while LM Studio jobs continue.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


DEFAULT_OUT_DIR = (
    "<WORKSPACE>/"
    "paper_assets/paper_a/rerun_logs/preference_250_extended_models_20260620"
)
DEFAULT_MODELS = (
    "google/gemma-4-12b google/gemma-4-e4b google/gemma-3-1b "
    "qwen/qwen3-1.7b qwenpaw-flash-9b"
)
DEFAULT_MODES = "language_to_cost route_option_rank"
EPISODE_RE = re.compile(r"^episode=(?P<episode>\d+)\b")
FIELD_RE = re.compile(r"(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)=(?P<value>'[^']*'|\"[^\"]*\"|[^ ]+)")


@dataclass
class JobStatus:
    mode: str
    model: str
    episode_csv: Path
    summary_csv: Path
    log_path: Path
    rows: int = 0
    source: str = "missing"
    updated_at: float = 0.0
    success_rate: float | None = None
    strict_rate: float | None = None
    parse_rate: float | None = None
    semantic_cost: float | None = None
    outcome: str = ""
    last_episode: int | None = None


def sanitize(value: str) -> str:
    return value.replace("/", "_").replace(":", "_").replace(" ", "_")


def split_words(text: str | None, fallback: str) -> list[str]:
    raw = text if text is not None and text.strip() else fallback
    return [part for part in raw.split() if part]


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--"
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d{hours:02d}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def short_model(model: str, width: int = 25) -> str:
    if len(model) <= width:
        return model
    if "/" in model:
        tail = model.split("/", 1)[1]
        if len(tail) <= width:
            return tail
    return model[: width - 1] + "…"


def numeric(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in {None, "", "nan", "None"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def parse_log_episode_line(line: str) -> dict[str, str] | None:
    if not EPISODE_RE.match(line):
        return None
    return {match.group("key"): match.group("value").strip("'\"") for match in FIELD_RE.finditer(line)}


def log_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parsed = parse_log_episode_line(line)
                if parsed:
                    rows.append(parsed)
    except Exception:
        return rows
    return rows


def rate_from_rows(rows: list[dict[str, str]], key: str, outcome_success: bool = False) -> float | None:
    if not rows:
        return None
    vals: list[float] = []
    for row in rows:
        if outcome_success:
            vals.append(1.0 if row.get("outcome") == "success" else 0.0)
            continue
        val = numeric(row, key)
        if val is not None:
            vals.append(val)
    if not vals:
        return None
    return mean(vals)


def mean_from_rows(rows: list[dict[str, str]], key: str) -> float | None:
    vals = [numeric(row, key) for row in rows]
    vals = [val for val in vals if val is not None]
    return mean(vals) if vals else None


def file_time(path: Path) -> float:
    if not path.exists():
        return 0.0
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def oldest_project_time(out_dir: Path) -> float | None:
    times: list[float] = []
    for path in out_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        times.append(getattr(stat, "st_birthtime", stat.st_mtime))
    return min(times) if times else None


def build_job(out_dir: Path, episodes: int, mode: str, model: str) -> JobStatus:
    label = f"language_cost_option_ppo_{mode}_{sanitize(model)}"
    episode_csv = out_dir / "episodes" / f"{label}_{episodes}ep.csv"
    summary_csv = out_dir / "summaries" / f"{label}_{episodes}ep.csv"
    log_path = out_dir / f"{label}_{episodes}ep.log"
    job = JobStatus(mode=mode, model=model, episode_csv=episode_csv, summary_csv=summary_csv, log_path=log_path)

    rows = csv_rows(episode_csv)
    if rows:
        job.rows = len(rows)
        job.source = "csv"
        job.updated_at = file_time(episode_csv)
        job.success_rate = rate_from_rows(rows, "success")
        if job.success_rate is None:
            job.success_rate = rate_from_rows(rows, "outcome", outcome_success=True)
        job.strict_rate = rate_from_rows(rows, "plan_valid")
        job.parse_rate = rate_from_rows(rows, "parse_ok")
        job.semantic_cost = mean_from_rows(rows, "semantic_cost")
        job.outcome = rows[-1].get("outcome", "")
        try:
            job.last_episode = int(rows[-1].get("episode", "")) if rows[-1].get("episode") else None
        except ValueError:
            job.last_episode = job.rows
        return job

    rows = log_rows(log_path)
    if rows:
        job.rows = min(len(rows), episodes)
        job.source = "log"
        job.updated_at = file_time(log_path)
        job.success_rate = rate_from_rows(rows, "outcome", outcome_success=True)
        job.strict_rate = rate_from_rows(rows, "plan_valid")
        job.parse_rate = rate_from_rows(rows, "parse_ok")
        job.semantic_cost = mean_from_rows(rows, "semantic_cost")
        job.outcome = rows[-1].get("outcome", "")
        try:
            job.last_episode = int(rows[-1].get("episode", "")) if rows[-1].get("episode") else None
        except ValueError:
            job.last_episode = job.rows
    return job


def color(text: str, code: str, enabled: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if enabled else text


def progress_bar(done: int, total: int, width: int = 42, color_enabled: bool = True) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    ratio = max(0.0, min(1.0, done / total))
    filled = int(round(width * ratio))
    bar = "█" * filled + "░" * (width - filled)
    code = "32" if ratio >= 1.0 else "36"
    return "[" + color(bar, code, color_enabled) + "]"


def percent(done: int, total: int) -> str:
    if total <= 0:
        return "  0.0%"
    return f"{done * 100.0 / total:5.1f}%"


def fmt_rate(value: float | None) -> str:
    return "  --" if value is None else f"{100.0 * value:4.0f}"


def fmt_mean(value: float | None) -> str:
    return "  --" if value is None else f"{value:4.3f}"


def job_state(
    job: JobStatus,
    episodes: int,
    newest_log: Path | None,
    active_runner: bool,
    color_enabled: bool,
) -> str:
    if job.rows >= episodes:
        return color("DONE", "32;1", color_enabled)
    if job.log_path == newest_log and job.rows > 0:
        return color("RUN ", "33;1", color_enabled) if active_runner else color("IDLE", "31;1", color_enabled)
    if job.rows > 0:
        return color("PART", "35;1", color_enabled)
    return color("WAIT", "90", color_enabled)


def discover_newest_log(jobs: list[JobStatus]) -> Path | None:
    logs = [job.log_path for job in jobs if job.log_path.exists()]
    if not logs:
        return None
    return max(logs, key=file_time)


def active_process_lines(out_dir: Path) -> list[str]:
    try:
        proc = subprocess.run(
            ["ps", "-axo", "pid=,etime=,pcpu=,pmem=,command="],
            check=False,
            text=True,
            capture_output=True,
        )
    except Exception:
        return []
    needle = str(out_dir)
    lines: list[str] = []
    for line in proc.stdout.splitlines():
        if needle not in line:
            continue
        if "watch_preference_progress" in line:
            continue
        lines.append(line.strip())
    return lines


def render(args: argparse.Namespace) -> str:
    out_dir = Path(args.out_dir).expanduser()
    models = split_words(args.models, DEFAULT_MODELS)
    modes = split_words(args.modes, DEFAULT_MODES)
    jobs = [build_job(out_dir, args.episodes_per_job, mode, model) for model in models for mode in modes]
    newest_log = discover_newest_log(jobs)
    color_enabled = (not args.no_color) and (sys.stdout.isatty() or args.force_color)

    total_episodes = len(jobs) * args.episodes_per_job
    done_episodes = sum(min(job.rows, args.episodes_per_job) for job in jobs)
    done_jobs = sum(1 for job in jobs if job.rows >= args.episodes_per_job)
    partial_jobs = sum(1 for job in jobs if 0 < job.rows < args.episodes_per_job)
    started_at = oldest_project_time(out_dir)
    now = time.time()
    elapsed = now - started_at if started_at else None
    rate = done_episodes / elapsed if elapsed and elapsed > 0 else 0.0
    eta = (total_episodes - done_episodes) / rate if rate > 0 and done_episodes < total_episodes else 0.0
    current = None
    if newest_log:
        matching = [job for job in jobs if job.log_path == newest_log]
        current = matching[0] if matching else None

    lines: list[str] = []
    title = "Paper A Preference Sweep Live Dashboard"
    lines.append(color(title, "1;36", color_enabled))
    lines.append(f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}   out: {out_dir}")
    lines.append("")
    lines.append(
        f"{progress_bar(done_episodes, total_episodes, color_enabled=color_enabled)} "
        f"{percent(done_episodes, total_episodes)}  "
        f"episodes {done_episodes}/{total_episodes}   jobs {done_jobs}/{len(jobs)} done"
    )
    lines.append(
        f"elapsed {fmt_duration(elapsed)}   eta {fmt_duration(eta)}   "
        f"speed {rate * 60.0:5.2f} ep/min   partial jobs {partial_jobs}"
    )
    if current:
        age = now - current.updated_at if current.updated_at else None
        lines.append(
            f"current: {color(current.mode, '33;1', color_enabled)} / "
            f"{color(current.model, '33;1', color_enabled)}   "
            f"{current.rows}/{args.episodes_per_job} ep   "
            f"last={current.outcome or '--'}   updated {fmt_duration(age)} ago"
        )
    else:
        lines.append("current: no active preference log detected yet")
    active = active_process_lines(out_dir)
    active_runner = bool(active)
    if active:
        lines.append(f"process: active ({len(active)} matching process line(s))")
        for proc_line in active[:2]:
            if len(proc_line) > 150:
                proc_line = proc_line[:149] + "…"
            lines.append(f"         {proc_line}")
    else:
        lines.append("process: no matching runner process found")
        if current and current.rows < args.episodes_per_job:
            lines.append(color("warning: latest job is incomplete and appears idle; resume with SKIP_COMPLETED=1.", "31;1", color_enabled))
    lines.append("")
    lines.append("state  mode                 model                     ep       succ strict parse sem    updated")
    lines.append("-----  -------------------  ------------------------  -------  ---- ------ ----- -----  --------")
    for job in jobs:
        updated = fmt_duration(now - job.updated_at) + " ago" if job.updated_at else "--"
        lines.append(
            f"{job_state(job, args.episodes_per_job, newest_log, active_runner, color_enabled):5}  "
            f"{job.mode[:19]:19}  "
            f"{short_model(job.model):24}  "
            f"{job.rows:3d}/{args.episodes_per_job:<3d}  "
            f"{fmt_rate(job.success_rate)}  {fmt_rate(job.strict_rate)}  "
            f"{fmt_rate(job.parse_rate)}  {fmt_mean(job.semantic_cost)}  "
            f"{updated:>8}"
        )
    lines.append("")
    lines.append("succ/strict/parse are percentages; sem is mean semantic cost. Ctrl+C exits monitor only.")
    if done_episodes >= total_episodes:
        lines.append(color("All expected preference jobs are complete.", "32;1", color_enabled))
        lines.append(f"summary target: {out_dir / 'language_cost_option_summary_with_ci.csv'}")
        lines.append(f"by-type target: {out_dir / 'preference_by_type_summary.csv'}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=os.environ.get("OUT_DIR", DEFAULT_OUT_DIR))
    parser.add_argument("--episodes-per-job", type=int, default=int(os.environ.get("EPISODES", "250")))
    parser.add_argument("--models", default=os.environ.get("MODEL_LIST", DEFAULT_MODELS))
    parser.add_argument("--modes", default=os.environ.get("LLM_BASELINE_MODES", DEFAULT_MODES))
    parser.add_argument("--refresh", type=float, default=float(os.environ.get("REFRESH_SECONDS", "5")))
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit.")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear the screen between refreshes.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    parser.add_argument("--force-color", action="store_true", help="Emit ANSI colors even when stdout is not a TTY.")
    args = parser.parse_args()

    while True:
        if not args.no_clear and not args.once:
            sys.stdout.write("\033[2J\033[H")
        print(render(args), flush=True)
        if args.once:
            break
        time.sleep(max(1.0, args.refresh))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nmonitor stopped; experiment process is untouched.")
