"""Stress test software-in-the-loop untuk Pulse Block XYZ-angle.

Model ini tidak menggantikan uji EMI/tegangan perangkat keras, tetapi menguji
timeline, kapasitas buffer, beban serial, kuantisasi pulsa Z, selip, stall, dan
kegagalan pembacaan encoder terhadap CSV nyata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from simulate_saved_trajectory import build_route, load_bundle

import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "full_gui"))
from pulse_planner import ISR_TICK_US, plan_polyline  # noqa: E402


TIMER_HZ = 1_000_000.0 / ISR_TICK_US
Z_STEPS_PER_DEG = 6400.0 * 20.0 / 360.0
ENCODER_COUNTS_PER_DEG = 4096.0 / 360.0


def make_plans(path, speed, z_deg_per_cm):
    meta, prefix, segments = load_bundle(path)
    route, actions = build_route(prefix, segments)
    checkpoints = sorted(actions)
    if len(route) - 1 not in checkpoints:
        checkpoints.append(len(route) - 1)
    offset = float(meta.get("grip_release_z_offset", 25.0))
    plans = []
    start = 0

    def add(points):
        points = np.asarray(points, dtype=float)
        if len(points) >= 2 and np.linalg.norm(np.diff(points, axis=0)) > 1e-9:
            plans.append(plan_polyline(points, max_speed_mm_s=speed, z_deg_per_cm=z_deg_per_cm))

    for checkpoint in checkpoints:
        add(route[start:checkpoint + 1])
        point = route[checkpoint]
        for _action in actions.get(checkpoint, []):
            action_point = point.copy()
            action_point[2] += offset
            add([point, action_point])
            add([action_point, point])
        start = checkpoint
    return plans


def all_z_targets(plans):
    return np.asarray(
        [slice_data[2] / 10.0 for plan in plans for block in plan.blocks for slice_data in block.slices],
        dtype=float,
    )


def serial_and_pulse_load(plans, speed, z_deg_per_cm):
    blocks = [block for plan in plans for block in plan.blocks]
    duration = sum(plan.duration_s for plan in plans)
    command_bytes = sum(len(block.command.encode("ascii")) for block in blocks)
    # ACK dan DONE aktual bervariasi panjang; 42 byte/blok adalah estimasi konservatif.
    serial_bytes = command_bytes + 42 * len(blocks)
    steps_per_mm = np.asarray([280.0, 140.0])
    pulse_xy = np.sum([np.sum(np.abs(plan.delta_steps), axis=0) for plan in plans], axis=0)
    requested_z_hz = speed[2] * (z_deg_per_cm / 10.0) * Z_STEPS_PER_DEG
    z_interval = max(2, int(TIMER_HZ / requested_z_hz + 0.5))
    actual_z_hz = TIMER_HZ / z_interval
    return {
        "blocks": len(blocks),
        "duration_s": duration,
        "serial_average_bytes_s": serial_bytes / duration,
        "serial_capacity_bytes_s_8n1": 250000.0 / 10.0,
        "serial_utilization_percent": serial_bytes / duration / 25000.0 * 100.0,
        "x_average_pulse_hz": pulse_xy[0] / duration,
        "y_average_pulse_hz": pulse_xy[1] / duration,
        "z_configured_pulse_hz": actual_z_hz,
        "z_interval_ticks": z_interval,
        "timer_isr_hz": TIMER_HZ,
        "x_calibration_steps_per_mm": steps_per_mm[0],
        "y_calibration_steps_per_mm": steps_per_mm[1],
    }


def simulate_z(targets, speed_z, z_deg_per_cm, *, slip=1.0, stall_at=None,
               permanent_encoder_failure_at=None, transient_failure_probability=0.0,
               seed=1):
    rng = np.random.default_rng(seed)
    requested_hz = speed_z * (z_deg_per_cm / 10.0) * Z_STEPS_PER_DEG
    interval = max(2, int(TIMER_HZ / requested_hz + 0.5))
    deg_per_slice = (TIMER_HZ / interval) / Z_STEPS_PER_DEG * 0.010 * slip
    actual = targets[0]
    measured = actual
    last_progress_slice = 0
    progress_reference = round(measured * ENCODER_COUNTS_PER_DEG)
    read_failures = 0
    max_error = 0.0
    outcome = "PASS"
    stopped_slice = None

    for index, target in enumerate(targets):
        stalled = stall_at is not None and index >= stall_at
        if not stalled:
            error = target - measured
            actual += np.sign(error) * min(abs(error), deg_per_slice)

        # Feedback encoder dibaca setiap 20 ms, yaitu satu sampel untuk dua
        # slice trajectory yang masing-masing berdurasi 10 ms.
        sample_due = index % 2 == 1
        if sample_due:
            failed = (
                permanent_encoder_failure_at is not None and index >= permanent_encoder_failure_at
            ) or rng.random() < transient_failure_probability
            if failed:
                read_failures += 1
                if read_failures >= 3:
                    outcome = "ERROR:Z_ENCODER_READ"
                    stopped_slice = index
                    break
            else:
                read_failures = 0
                measured = round(actual * ENCODER_COUNTS_PER_DEG) / ENCODER_COUNTS_PER_DEG

            count = round(measured * ENCODER_COUNTS_PER_DEG)
            if abs(count - progress_reference) >= 2:
                progress_reference = count
                last_progress_slice = index
        if abs(target - measured) > 0.5 and index - last_progress_slice >= 300:
            outcome = "ERROR:Z_STALL"
            stopped_slice = index
            break
        max_error = max(max_error, abs(target - measured))

    # Firmware menahan STREAM_DONE setelah slice terakhir sampai Z masuk
    # toleransi. Simulasikan fase catch-up maksimal 30 detik.
    if outcome == "PASS":
        final_target = targets[-1]
        for extra in range(3000):
            if abs(final_target - measured) <= 0.5:
                break
            actual += np.sign(final_target - measured) * min(
                abs(final_target - measured), deg_per_slice
            )
            measured = round(actual * ENCODER_COUNTS_PER_DEG) / ENCODER_COUNTS_PER_DEG
        if abs(final_target - measured) > 0.5:
            outcome = "ERROR:Z_FINAL_TIMEOUT"

    return {
        "outcome": outcome,
        "stopped_at_s": None if stopped_slice is None else stopped_slice * 0.01,
        "max_tracking_error_deg": max_error,
        "final_error_deg": abs(targets[min(len(targets) - 1, stopped_slice or len(targets) - 1)] - measured),
    }


def simulate_buffer(total_blocks, *, host_pause_every=500, host_pause_s=1.2):
    # Buffer awal 32 blok = 640 ms. BUFFER_WAIT bersifat pause/resume, bukan gagal.
    pause_events = (total_blocks - 1) // host_pause_every if host_pause_every else 0
    buffer_seconds = 32 * 0.020
    underruns = pause_events if host_pause_s > buffer_seconds else 0
    wait_s = underruns * max(0.0, host_pause_s - buffer_seconds)
    return {
        "injected_host_pause_events": pause_events,
        "buffer_wait_events": underruns,
        "recoverable_wait_total_s": wait_s,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--speed-x", type=float, default=80.0)
    parser.add_argument("--speed-y", type=float, default=80.0)
    parser.add_argument("--speed-z", type=float, default=20.0)
    parser.add_argument("--z-deg-per-cm", type=float, default=15.0)
    args = parser.parse_args()
    speed = (args.speed_x, args.speed_y, args.speed_z)
    plans = make_plans(args.csv_path, speed, args.z_deg_per_cm)
    targets = all_z_targets(plans)
    middle = len(targets) // 2
    load = serial_and_pulse_load(plans, speed, args.z_deg_per_cm)
    scenarios = {
        "nominal": simulate_z(targets, args.speed_z, args.z_deg_per_cm),
        "z_slip_20_percent": simulate_z(targets, args.speed_z, args.z_deg_per_cm, slip=0.8),
        "z_slip_50_percent": simulate_z(targets, args.speed_z, args.z_deg_per_cm, slip=0.5),
        "transient_i2c_0_01_percent": simulate_z(
            targets, args.speed_z, args.z_deg_per_cm,
            transient_failure_probability=0.0001, seed=22,
        ),
        "permanent_encoder_failure": simulate_z(
            targets, args.speed_z, args.z_deg_per_cm,
            permanent_encoder_failure_at=middle,
        ),
        "mechanical_z_stall": simulate_z(
            targets, args.speed_z, args.z_deg_per_cm, stall_at=middle,
        ),
    }
    result = {
        "source": str(args.csv_path),
        "load": load,
        "buffer_with_1_2s_host_pause_every_500_blocks": simulate_buffer(
            load["blocks"], host_pause_every=500, host_pause_s=1.2
        ),
        "scenarios": scenarios,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
