"""Simulasi offline CSV Segment Planner pada planner Pulse Block XYZ-angle."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "full_gui"))

from pulse_planner import ISR_TICK_US, plan_polyline  # noqa: E402


def _point(row):
    return np.asarray([float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])])


def load_bundle(path: Path):
    prefix = []
    segments = defaultdict(lambda: {"outbound": [], "return": []})
    meta = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            kind = (row.get("record_type") or "").upper()
            if kind == "META":
                meta = json.loads(row.get("payload_json") or "{}")
            elif kind == "PREFIX":
                prefix.append(_point(row))
            elif kind in ("SEGMENT_OUTBOUND", "SEGMENT_RETURN"):
                index = int(row.get("group_index") or 0)
                key = "outbound" if kind.endswith("OUTBOUND") else "return"
                segments[index][key].append(_point(row))
    if not segments:
        raise ValueError("CSV tidak memiliki SEGMENT_OUTBOUND/SEGMENT_RETURN.")
    return meta, prefix, [segments[index] for index in sorted(segments)]


def append_unique(route, point):
    point = np.asarray(point, dtype=float)
    if not route or np.linalg.norm(route[-1] - point) > 1e-6:
        route.append(point)
    return len(route) - 1


def build_route(prefix, segments):
    route, actions = [], {}
    for point in prefix:
        append_unique(route, point)
    if route:
        actions.setdefault(len(route) - 1, []).append("GRIP")
    for index, segment in enumerate(segments):
        for point in segment["outbound"]:
            append_unique(route, point)
        actions.setdefault(len(route) - 1, []).append("RELEASE")
        if index < len(segments) - 1:
            for point in segment["return"]:
                append_unique(route, point)
            actions.setdefault(len(route) - 1, []).append("GRIP")
    return np.asarray(route), actions


def simulate_z(plan, speed_z_mm_s, z_deg_per_cm):
    # Model ideal firmware: interval timer integer, bang-bang, toleransi 0,5°.
    timer_hz = 1_000_000.0 / ISR_TICK_US
    steps_per_degree = 6400.0 * 20.0 / 360.0
    requested_pulse_hz = speed_z_mm_s * (z_deg_per_cm / 10.0) * steps_per_degree
    interval_ticks = max(2, int(timer_hz / requested_pulse_hz + 0.5))
    pulse_hz = timer_hz / interval_ticks
    degrees_per_slice = pulse_hz / steps_per_degree * (plan.slice_us / 1_000_000.0)
    actual = float(plan.blocks[0].slices[0][2]) / 10.0
    max_lag = 0.0
    for block in plan.blocks:
        for _dx, _dy, target_deci in block.slices:
            target = target_deci / 10.0
            error = target - actual
            actual += np.sign(error) * min(abs(error), degrees_per_slice)
            max_lag = max(max_lag, abs(target - actual))
    final_target = plan.blocks[-1].slices[-1][2] / 10.0
    catchup_s = max(0.0, abs(final_target - actual) / (pulse_hz / steps_per_degree))
    return {
        "interval_ticks": interval_ticks,
        "actual_speed_mm_s": pulse_hz / steps_per_degree / (z_deg_per_cm / 10.0),
        "max_lag_deg": max_lag,
        "final_error_before_wait_deg": abs(final_target - actual),
        "final_catchup_s": catchup_s,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--speed-x", type=float, default=80.0)
    parser.add_argument("--speed-y", type=float, default=80.0)
    parser.add_argument("--speed-z", type=float, default=20.0)
    parser.add_argument("--z-deg-per-cm", type=float, default=15.0)
    args = parser.parse_args()

    meta, prefix, segments = load_bundle(args.csv_path)
    route, actions = build_route(prefix, segments)
    checkpoints = sorted(actions)
    if route.shape[0] - 1 not in checkpoints:
        checkpoints.append(route.shape[0] - 1)

    offset = float(meta.get("grip_release_z_offset", 25.0))
    start = 0
    rows = []
    total_blocks = total_slices = 0
    total_duration = total_catchup = 0.0
    max_dx = max_dy = max_z_jump = 0
    max_z_lag = 0.0

    def analyze(label, points):
        nonlocal total_blocks, total_slices, total_duration, total_catchup
        nonlocal max_dx, max_dy, max_z_jump, max_z_lag
        plan = plan_polyline(
            np.asarray(points),
            max_speed_mm_s=(args.speed_x, args.speed_y, args.speed_z),
            z_deg_per_cm=args.z_deg_per_cm,
            slice_ms=10,
            block_ms=20,
        )
        z_targets = np.asarray([s[2] for b in plan.blocks for s in b.slices], dtype=int)
        z_sim = simulate_z(plan, args.speed_z, args.z_deg_per_cm)
        max_dx = max(max_dx, int(np.max(np.abs(plan.delta_steps[:, 0]))))
        max_dy = max(max_dy, int(np.max(np.abs(plan.delta_steps[:, 1]))))
        max_z_jump = max(max_z_jump, int(np.max(np.abs(np.diff(z_targets))))) if len(z_targets) > 1 else max_z_jump
        max_z_lag = max(max_z_lag, z_sim["max_lag_deg"])
        total_blocks += len(plan.blocks)
        total_slices += len(plan.delta_steps)
        total_duration += plan.duration_s
        total_catchup += z_sim["final_catchup_s"]
        rows.append({
            "label": label,
            "points": len(points),
            "blocks": len(plan.blocks),
            "duration_s": round(plan.duration_s, 3),
            **{key: round(value, 4) if isinstance(value, float) else value for key, value in z_sim.items()},
        })

    for checkpoint in checkpoints:
        chunk = route[start:checkpoint + 1]
        if len(chunk) >= 2:
            analyze(f"route_{start}_{checkpoint}", chunk)
        point = route[checkpoint]
        for action in actions.get(checkpoint, []):
            action_point = point.copy()
            action_point[2] += offset
            analyze(f"{action}_offset_up", [point, action_point])
            analyze(f"{action}_offset_down", [action_point, point])
        start = checkpoint

    result = {
        "source": str(args.csv_path),
        "route_points": len(route),
        "actions": sum(len(value) for value in actions.values()),
        "chunks": len(rows),
        "total_blocks": total_blocks,
        "total_slices": total_slices,
        "planned_duration_s": round(total_duration, 3),
        "z_final_wait_total_s": round(total_catchup, 3),
        "max_abs_dx_steps_per_slice": max_dx,
        "max_abs_dy_steps_per_slice": max_dy,
        "max_z_target_jump_deci_deg": max_z_jump,
        "max_z_tracking_lag_deg": round(max_z_lag, 4),
        "timer_ticks_per_slice": 500,
        "details": rows,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
