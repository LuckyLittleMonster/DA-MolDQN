#!/usr/bin/env python
"""
Training Monitor Dashboard - real-time parsing of training logs.

Auto-discovers active training logs in Experiments/logs/ and displays
a unified progress dashboard with key metrics.

Supported log formats:
  - V3 Link Predictor: Epoch X/Y | Train: loss=... | Val: AUC=..., AP=... | lr=... | Xs
  - AIO Directed Hypergraph: Epoch X/Y | Train: ... (prod:... co:...) | Val: ... Acc:...% | lr=... | Xs
  - RL DQN Training: Episode X/Y | Reward: ... | QED: ... | SA: ... | Eps: ... | Time: ...s

Usage:
    # One-shot dashboard
    python scripts/monitor_training.py

    # Watch mode (refresh every 30s)
    python scripts/monitor_training.py --watch

    # Watch with custom interval
    python scripts/monitor_training.py --watch --interval 60

    # Only show active (recently modified) logs
    python scripts/monitor_training.py --active-only
"""

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EpochInfo:
    epoch: int
    total_epochs: int
    train_loss: float = 0.0
    val_metric: float = 0.0
    val_metric_name: str = ""
    lr: float = 0.0
    epoch_time: float = 0.0
    is_best: bool = False
    extra: Dict = field(default_factory=dict)


@dataclass
class TrainingRun:
    name: str
    log_path: str
    model_type: str  # "v3", "aio", "rl"
    epochs: List[EpochInfo] = field(default_factory=list)
    job_id: str = ""
    node: str = ""
    start_time: str = ""
    config_info: str = ""
    total_epochs: int = 0
    best_val: float = 0.0
    best_epoch: int = 0
    last_modified: float = 0.0
    status: str = "unknown"  # "running", "completed", "stalled", "preparing"


# =============================================================================
# Log Parsers
# =============================================================================

# V3 epoch: Epoch   1/60 | Train: loss=0.9651 (bce=0.451, ctr=0.858, rxn=0.283) | Val: AUC=0.9628, AP=0.9714, Acc=0.7956, F1=0.7439 | lr=3.40e-06 | 571.0s
RE_V3_EPOCH = re.compile(
    r'Epoch\s+(\d+)/(\d+)\s+\|'
    r'\s+Train:\s+loss=([\d.]+)\s*'
    r'(?:\(([^)]*)\))?\s*\|'
    r'\s+Val:\s+AUC=([\d.]+),\s*AP=([\d.]+),\s*Acc=([\d.]+),\s*F1=([\d.]+)'
    r'\s+\|\s+lr=([\d.eE+-]+)'
    r'\s+\|\s+([\d.]+)s'
)

# AIO epoch: Epoch   1/60 | Train: 2.4457 (prod:2.340 co:0.108 dir:0.003 cls:0.038) | Val: 1.4705 Acc:100.0% | lr=6.80e-05 | 394.3s
RE_AIO_EPOCH = re.compile(
    r'Epoch\s+(\d+)/(\d+)\s+\|'
    r'\s+Train:\s+([\d.]+)\s*'
    r'(?:\(([^)]*)\))?\s*\|'
    r'\s+Val:\s+([\d.]+)\s+Acc:([\d.]+)%'
    r'\s+\|\s+lr=([\d.eE+-]+)'
    r'\s+\|\s+([\d.]+)s'
)

# RL episode format 1: Episode 10/2000 | Reward: 0.5018 | QED: 0.441 | SA: 3.28 | Eps: 0.923 | Time: 19.89s
RE_RL_EPISODE = re.compile(
    r'Episode\s+(\d+)/(\d+)\s+\|'
    r'\s+Reward:\s+([\d.]+)\s+\|'
    r'\s+QED:\s+([\d.]+)\s+\|'
    r'\s+SA:\s+([\d.]+)\s+\|'
    r'\s+Eps:\s+([\d.]+)\s+\|'
    r'\s+Time:\s+([\d.]+)s'
)

# RL episode format 2: Ep    50/8000 | QED: 0.119 | Avg100: 0.294 | Best: 0.446 | Eps: 0.9522 | Loss: 0.0467 | Mol: ... | Time: 0.44s/ep
RE_RL_EP_V2 = re.compile(
    r'Ep\s+(\d+)/(\d+)\s+\|'
    r'\s+QED:\s+([\d.]+)\s+\|'
    r'\s+Avg100:\s+([\d.]+)\s+\|'
    r'\s+Best:\s+([\d.]+)\s+\|'
    r'\s+Eps:\s+([\d.]+)\s+\|'
    r'\s+Loss:\s+([\d.]+)\s+\|'
    r'\s+Mol:.*\|'
    r'\s+Time:\s+([\d.]+)s/ep'
)

RE_SAVED_BEST = re.compile(r'Saved best model|-> Saved best')
RE_JOB_ID = re.compile(r'Job ID:\s*(\d+)')
RE_NODE = re.compile(r'Node:\s*([\w.-]+)')
RE_START_TIME = re.compile(r'Start time:\s*(.*)')
RE_TRAINING_FOR = re.compile(r'Training(?::\s+|\s+for\s+)(\d+)\s+(?:episodes|epochs)')
RE_EARLY_STOP = re.compile(r'early stop patience=(\d+)')
RE_EARLY_STOPPED = re.compile(r'Early stopping at epoch (\d+)')
RE_TEST_RESULTS = re.compile(r'roc_auc:\s+([\d.]+)')


def detect_model_type(lines: List[str], filename: str = "") -> str:
    """Detect log type from content and filename."""
    text = "\n".join(lines[:200])
    if "Val: AUC=" in text:
        return "v3"
    if "prod:" in text and "co:" in text:
        return "aio"
    if ("Episode" in text and "QED:" in text) or ("Ep " in text and "Avg100:" in text):
        return "rl"
    # Content-based heuristics for preparation stage logs
    if "CF-NCN" in text or "NCN:" in text or "Link Predictor" in text:
        return "v3"
    if "DirectedHypergraph" in text or "Directed Hypergraph" in text:
        return "aio"
    if "ReactionPredictor" in text or "episodes" in text.lower():
        return "rl"
    # Filename fallback
    fname = filename.lower()
    if "v3" in fname or "ddp" in fname:
        return "v3"
    if "aio" in fname:
        return "aio"
    if "qed" in fname or "batch" in fname or "gnn" in fname or "mlp" in fname:
        return "rl"
    return "unknown"


def parse_log(log_path: str) -> TrainingRun:
    """Parse a training log file and extract all epoch/episode info."""
    name = Path(log_path).stem
    run = TrainingRun(name=name, log_path=log_path, model_type="unknown")
    run.last_modified = os.path.getmtime(log_path)

    try:
        with open(log_path, "r", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        run.status = "error"
        return run

    if not lines:
        run.status = "empty"
        return run

    # Also check .out companion if this is .err
    companion_lines = []
    if log_path.endswith(".err"):
        out_path = log_path[:-4] + ".out"
        if os.path.exists(out_path):
            try:
                with open(out_path, "r", errors="replace") as f:
                    companion_lines = f.readlines()
                run.last_modified = max(run.last_modified, os.path.getmtime(out_path))
            except Exception:
                pass
    elif log_path.endswith(".out"):
        err_path = log_path[:-4] + ".err"
        if os.path.exists(err_path):
            run.last_modified = max(run.last_modified, os.path.getmtime(err_path))

    all_lines = companion_lines + lines if companion_lines else lines

    # Detect model type
    run.model_type = detect_model_type(all_lines, name)

    # Extract metadata
    for line in all_lines[:80]:
        m = RE_JOB_ID.search(line)
        if m:
            run.job_id = m.group(1)
        m = RE_NODE.search(line)
        if m:
            run.node = m.group(1)
        m = RE_START_TIME.search(line)
        if m:
            run.start_time = m.group(1).strip()
        m = RE_TRAINING_FOR.search(line)
        if m:
            run.total_epochs = int(m.group(1))

    # Extract config info from header
    for line in all_lines[:10]:
        line = line.strip()
        if line and not line.startswith("=") and "Job ID" not in line:
            if "lr=" in line or "LR" in line.upper() or "warmup" in line.lower():
                run.config_info = line
                break

    # Parse epochs/episodes
    best_lines = set()
    for i, line in enumerate(all_lines):
        if RE_SAVED_BEST.search(line):
            best_lines.add(i - 1)  # The epoch line is usually right before

    for i, line in enumerate(all_lines):
        line = line.strip()

        # V3 format
        m = RE_V3_EPOCH.match(line)
        if m:
            ep = EpochInfo(
                epoch=int(m.group(1)),
                total_epochs=int(m.group(2)),
                train_loss=float(m.group(3)),
                val_metric=float(m.group(5)),  # AUC
                val_metric_name="AUC",
                lr=float(m.group(9)),
                epoch_time=float(m.group(10)),
                is_best=(i in best_lines or i + 1 in [j + 1 for j in best_lines]),
                extra={
                    "loss_parts": m.group(4) or "",
                    "ap": float(m.group(6)),
                    "acc": float(m.group(7)),
                    "f1": float(m.group(8)),
                },
            )
            run.epochs.append(ep)
            run.total_epochs = max(run.total_epochs, ep.total_epochs)
            continue

        # AIO format
        m = RE_AIO_EPOCH.match(line)
        if m:
            val_loss = float(m.group(5))
            ep = EpochInfo(
                epoch=int(m.group(1)),
                total_epochs=int(m.group(2)),
                train_loss=float(m.group(3)),
                val_metric=val_loss,
                val_metric_name="Val Loss",
                lr=float(m.group(7)),
                epoch_time=float(m.group(8)),
                is_best=(i in best_lines or i + 1 in [j + 1 for j in best_lines]),
                extra={
                    "loss_parts": m.group(4) or "",
                    "acc": float(m.group(6)),
                },
            )
            run.epochs.append(ep)
            run.total_epochs = max(run.total_epochs, ep.total_epochs)
            continue

        # RL format 1
        m = RE_RL_EPISODE.match(line)
        if m:
            ep = EpochInfo(
                epoch=int(m.group(1)),
                total_epochs=int(m.group(2)),
                train_loss=float(m.group(3)),  # reward
                val_metric=float(m.group(4)),  # QED
                val_metric_name="QED",
                lr=float(m.group(6)),  # epsilon
                epoch_time=float(m.group(7)),
                extra={"sa": float(m.group(5))},
            )
            run.epochs.append(ep)
            run.total_epochs = max(run.total_epochs, ep.total_epochs)
            continue

        # RL format 2 (gnn/mlp verify)
        m = RE_RL_EP_V2.match(line)
        if m:
            ep = EpochInfo(
                epoch=int(m.group(1)),
                total_epochs=int(m.group(2)),
                train_loss=float(m.group(7)),  # loss
                val_metric=float(m.group(3)),  # QED (current)
                val_metric_name="QED",
                lr=float(m.group(6)),  # epsilon
                epoch_time=float(m.group(8)),
                extra={
                    "avg100": float(m.group(4)),
                    "best": float(m.group(5)),
                },
            )
            run.epochs.append(ep)
            run.total_epochs = max(run.total_epochs, ep.total_epochs)
            continue

    # Check for best saved markers that appear on their own line after epoch
    for i, line in enumerate(all_lines):
        if RE_SAVED_BEST.search(line):
            # Mark the most recent epoch as best
            if run.epochs:
                run.epochs[-1].is_best = True

    # Compute best val metric
    if run.epochs:
        if run.model_type == "aio":
            # For AIO, lower val loss is better
            best_ep = min(run.epochs, key=lambda e: e.val_metric)
        else:
            # For V3/RL, higher is better
            best_ep = max(run.epochs, key=lambda e: e.val_metric)
        run.best_val = best_ep.val_metric
        run.best_epoch = best_ep.epoch

    # Check for early stopping / completion markers in text
    early_stopped = False
    test_auc = None
    for line in all_lines:
        m = RE_EARLY_STOPPED.search(line)
        if m:
            early_stopped = True
        m = RE_TEST_RESULTS.search(line)
        if m:
            test_auc = float(m.group(1))

    if test_auc is not None:
        run.config_info += f" [Test AUC={test_auc:.4f}]" if run.config_info else f"Test AUC={test_auc:.4f}"

    # Determine status
    age_hours = (time.time() - run.last_modified) / 3600
    if not run.epochs:
        if age_hours > 2.0:
            run.status = "stalled"
        else:
            run.status = "preparing"
    elif early_stopped or test_auc is not None:
        run.status = "completed"
    elif run.epochs[-1].epoch >= run.total_epochs and run.total_epochs > 0:
        run.status = "completed"
    elif age_hours > 1.0:
        run.status = "stalled"
    else:
        run.status = "running"

    return run


# =============================================================================
# Dashboard Rendering
# =============================================================================

def detect_early_stopping(run: TrainingRun, patience: int = 10) -> Optional[str]:
    """Check if training shows signs of early stopping."""
    if len(run.epochs) < patience:
        return None

    if run.model_type == "aio":
        # Lower is better for val loss
        best_val = min(e.val_metric for e in run.epochs)
        no_improve = 0
        for ep in reversed(run.epochs):
            if ep.val_metric > best_val * 1.001:  # slight tolerance
                no_improve += 1
            else:
                break
    else:
        # Higher is better
        best_val = max(e.val_metric for e in run.epochs)
        no_improve = 0
        for ep in reversed(run.epochs):
            if ep.val_metric < best_val * 0.999:
                no_improve += 1
            else:
                break

    if no_improve >= patience:
        return f"No improvement for {no_improve} epochs (patience={patience})"
    elif no_improve >= patience // 2:
        return f"Warning: {no_improve}/{patience} epochs without improvement"
    return None


def format_time(seconds: float) -> str:
    """Format seconds to human-readable."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    else:
        return f"{seconds / 86400:.1f}d"


def format_eta(run: TrainingRun) -> str:
    """Estimate remaining time."""
    if not run.epochs or run.total_epochs == 0:
        return "N/A"
    current = run.epochs[-1].epoch
    remaining = run.total_epochs - current
    if remaining <= 0:
        return "done"
    avg_time = sum(e.epoch_time for e in run.epochs) / len(run.epochs)
    eta_seconds = remaining * avg_time
    return format_time(eta_seconds)


def status_indicator(status: str) -> str:
    """Text-based status indicator."""
    return {
        "running": "[RUNNING]",
        "completed": "[DONE]",
        "stalled": "[STALLED]",
        "preparing": "[PREP]",
        "error": "[ERROR]",
        "empty": "[EMPTY]",
        "unknown": "[?]",
    }.get(status, f"[{status}]")


def render_dashboard(runs: List[TrainingRun], show_all: bool = False) -> str:
    """Render the training dashboard as markdown."""
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"# Training Monitor Dashboard")
    lines.append(f"Updated: {now}")
    lines.append("")

    # Separate by type - only show runs with epoch data in main tables
    v3_runs = [r for r in runs if r.model_type == "v3" and r.epochs]
    aio_runs = [r for r in runs if r.model_type == "aio" and r.epochs]
    rl_runs = [r for r in runs if r.model_type == "rl" and r.epochs]

    # V3 Link Predictor table
    if v3_runs:
        lines.append("## V3 Link Predictor Training")
        lines.append("")
        lines.append("| Name | Status | Epoch | Best AUC | Best Ep | Cur LR | Avg Time | ETA | Config |")
        lines.append("|:-----|:------:|------:|--------:|--------:|-------:|---------:|----:|:-------|")

        for run in sorted(v3_runs, key=lambda r: r.name):
            if run.epochs:
                last = run.epochs[-1]
                avg_t = sum(e.epoch_time for e in run.epochs) / len(run.epochs)
                eta = format_eta(run)
                lines.append(
                    f"| {run.name} | {status_indicator(run.status)} | "
                    f"{last.epoch}/{run.total_epochs} | "
                    f"{run.best_val:.4f} | {run.best_epoch} | "
                    f"{last.lr:.2e} | {format_time(avg_t)} | {eta} | "
                    f"{run.config_info[:40]} |"
                )
            else:
                lines.append(
                    f"| {run.name} | {status_indicator(run.status)} | "
                    f"0/{run.total_epochs} | -- | -- | -- | -- | -- | "
                    f"{run.config_info[:40]} |"
                )

        # Early stopping warnings
        for run in v3_runs:
            warning = detect_early_stopping(run)
            if warning:
                lines.append(f"  * {run.name}: {warning}")

        lines.append("")

        # Best epoch details
        lines.append("### Latest Epoch Details")
        lines.append("")
        for run in sorted(v3_runs, key=lambda r: r.name):
            if run.epochs:
                last = run.epochs[-1]
                extra = last.extra
                lines.append(
                    f"- **{run.name}** (ep {last.epoch}): "
                    f"loss={last.train_loss:.4f} [{extra.get('loss_parts', '')}] | "
                    f"AUC={last.val_metric:.4f}, AP={extra.get('ap', 0):.4f}, "
                    f"F1={extra.get('f1', 0):.4f}"
                )
        lines.append("")

    # AIO table
    if aio_runs:
        lines.append("## AIO Directed Hypergraph Training")
        lines.append("")
        lines.append("| Name | Status | Epoch | Best Val Loss | Best Ep | Cur LR | Avg Time | ETA |")
        lines.append("|:-----|:------:|------:|--------------:|--------:|-------:|---------:|----:|")

        for run in sorted(aio_runs, key=lambda r: r.name):
            if run.epochs:
                last = run.epochs[-1]
                avg_t = sum(e.epoch_time for e in run.epochs) / len(run.epochs)
                eta = format_eta(run)
                lines.append(
                    f"| {run.name} | {status_indicator(run.status)} | "
                    f"{last.epoch}/{run.total_epochs} | "
                    f"{run.best_val:.4f} | {run.best_epoch} | "
                    f"{last.lr:.2e} | {format_time(avg_t)} | {eta} |"
                )

        for run in aio_runs:
            warning = detect_early_stopping(run, patience=15)
            if warning:
                lines.append(f"  * {run.name}: {warning}")
        lines.append("")

        # Loss breakdown
        lines.append("### AIO Loss Breakdown")
        lines.append("")
        for run in sorted(aio_runs, key=lambda r: r.name):
            if run.epochs:
                last = run.epochs[-1]
                lines.append(
                    f"- **{run.name}** (ep {last.epoch}): "
                    f"total={last.train_loss:.4f} [{last.extra.get('loss_parts', '')}] | "
                    f"val={last.val_metric:.4f}, acc={last.extra.get('acc', 0):.1f}%"
                )
        lines.append("")

    # RL table
    if rl_runs:
        lines.append("## RL DQN Training")
        lines.append("")
        lines.append("| Name | Status | Episode | Best QED | Cur Eps | Avg Metric | Avg Time | ETA |")
        lines.append("|:-----|:------:|--------:|--------:|-------:|-----------:|---------:|----:|")

        for run in sorted(rl_runs, key=lambda r: r.name):
            if run.epochs:
                last = run.epochs[-1]
                avg_t = sum(e.epoch_time for e in run.epochs) / len(run.epochs)
                # Use avg100 if available (V2 format), otherwise use reward average
                if "avg100" in last.extra:
                    avg_metric = last.extra["avg100"]
                else:
                    avg_metric = sum(e.train_loss for e in run.epochs[-20:]) / min(20, len(run.epochs))
                eta = format_eta(run)
                lines.append(
                    f"| {run.name} | {status_indicator(run.status)} | "
                    f"{last.epoch}/{run.total_epochs} | "
                    f"{run.best_val:.3f} | {last.lr:.3f} | "
                    f"{avg_metric:.3f} | {format_time(avg_t)} | {eta} |"
                )
        lines.append("")

    # Preparing / stalled / other runs
    other_all = [r for r in runs if r.model_type not in ("v3", "aio", "rl") or not r.epochs]
    active_other = [r for r in other_all if r.status in ("preparing", "running")]
    stalled_other = [r for r in other_all if r.status == "stalled"]

    if active_other:
        lines.append("## Preparing (No Epochs Yet)")
        lines.append("")
        for run in active_other:
            age = format_time(time.time() - run.last_modified)
            lines.append(f"- {run.name}: [{run.model_type}] last updated {age} ago")
            if run.node:
                lines.append(f"  node={run.node}, job={run.job_id}")
        lines.append("")

    if stalled_other and show_all:
        lines.append("## Stalled / Old Runs")
        lines.append("")
        for run in stalled_other:
            age = format_time(time.time() - run.last_modified)
            lines.append(f"- {run.name}: [{run.model_type}] {len(run.epochs)} epochs, last updated {age} ago")
        lines.append("")

    # Summary
    total_running = sum(1 for r in runs if r.status == "running")
    total_done = sum(1 for r in runs if r.status == "completed")
    total_stalled = sum(1 for r in runs if r.status == "stalled")
    total_prep = sum(1 for r in runs if r.status == "preparing")

    lines.append("---")
    lines.append(f"Running: {total_running} | Completed: {total_done} | "
                 f"Stalled: {total_stalled} | Preparing: {total_prep}")

    return "\n".join(lines)


# =============================================================================
# Log Discovery
# =============================================================================

def discover_logs(log_dir: str, active_hours: float = 24.0) -> List[str]:
    """Discover training log files, preferring .out files."""
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return []

    # Collect all .out files
    out_files = sorted(log_dir.glob("*.out"))

    # Filter by recency if requested
    cutoff = time.time() - active_hours * 3600
    results = []
    for f in out_files:
        mtime = f.stat().st_mtime
        # Also check companion .err file
        err_f = f.with_suffix(".err")
        if err_f.exists():
            mtime = max(mtime, err_f.stat().st_mtime)
        if mtime >= cutoff:
            results.append(str(f))

    return results


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Training Monitor Dashboard")
    p.add_argument("--log-dir", type=str, default="Experiments/logs",
                   help="Log directory to monitor")
    p.add_argument("--watch", action="store_true",
                   help="Watch mode: refresh periodically")
    p.add_argument("--interval", type=int, default=30,
                   help="Refresh interval in seconds (with --watch)")
    p.add_argument("--active-only", action="store_true",
                   help="Only show logs modified in last 24h")
    p.add_argument("--active-hours", type=float, default=48.0,
                   help="Hours to consider a log active (default 48)")
    p.add_argument("--all", action="store_true",
                   help="Show all runs including those without epochs")
    p.add_argument("--output", "-o", type=str, default=None,
                   help="Save dashboard to file")
    return p.parse_args()


def main():
    args = parse_args()

    while True:
        # Discover logs
        if args.active_only:
            log_files = discover_logs(args.log_dir, args.active_hours)
        else:
            log_files = discover_logs(args.log_dir, 9999)

        if not log_files:
            print(f"No log files found in {args.log_dir}")
            if args.watch:
                time.sleep(args.interval)
                continue
            return

        # Parse all logs
        runs = []
        for lf in log_files:
            run = parse_log(lf)
            runs.append(run)

        # Render dashboard
        dashboard = render_dashboard(runs, show_all=args.all)

        if args.watch:
            # Clear screen
            os.system('clear' if os.name != 'nt' else 'cls')

        print(dashboard)

        if args.output:
            with open(args.output, "w") as f:
                f.write(dashboard + "\n")

        if not args.watch:
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
