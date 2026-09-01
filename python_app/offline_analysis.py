"""Simulasi ISR firmware Pulse Block Streaming tanpa Arduino."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pulse_planner import ISR_TICK_US, plan_sinusoidal


def simulate_sine(
    axis: str = "X",
    amplitude_mm: float = 20.0,
    period_s: float = 4.0,
    cycles: float = 2.0,
    steps_per_cm=(2800.0, 1400.0),
    slice_ms: int = 10,
    block_ms: int = 20,
    observation_ms: int = 1,
):
    plan = plan_sinusoidal(
        axis=axis,
        amplitude_mm=amplitude_mm,
        period_s=period_s,
        cycles=cycles,
        steps_per_cm=steps_per_cm,
        slice_ms=slice_ms,
        block_ms=block_ms,
    )
    axis_index = {"X": 0, "Y": 1}[axis.upper()]
    steps_per_mm = float(plan.steps_per_mm[axis_index])
    ticks_per_slice = plan.slice_us // ISR_TICK_US
    observe_ticks = max(1, observation_ms * 1000 // ISR_TICK_US)
    step_position = 0
    elapsed_ticks = 0
    samples_t = [0.0]
    samples_position = [0.0]

    for delta in plan.delta_steps[:, axis_index]:
        direction = (int(delta) > 0) - (int(delta) < 0)
        count = abs(int(delta))
        phase = 0
        for _ in range(ticks_per_slice):
            phase += count
            if phase >= ticks_per_slice:
                phase -= ticks_per_slice
                step_position += direction
            elapsed_ticks += 1
            if elapsed_ticks % observe_ticks == 0:
                samples_t.append(elapsed_ticks * ISR_TICK_US / 1_000_000.0)
                samples_position.append(step_position / steps_per_mm)

    t = np.asarray(samples_t)
    output = np.asarray(samples_position)
    effective_period = plan.duration_s / cycles
    target = amplitude_mm * np.sin(2.0 * np.pi * t / effective_period)
    error_um = (output - target) * 1000.0

    slice_center = (plan.times_s[:-1] + plan.times_s[1:]) * 0.5
    output_velocity = (
        plan.delta_steps[:, axis_index].astype(float)
        / steps_per_mm
        / (plan.slice_us / 1_000_000.0)
    )
    target_velocity = (
        amplitude_mm
        * (2.0 * np.pi / effective_period)
        * np.cos(2.0 * np.pi * slice_center / effective_period)
    )
    velocity_error = output_velocity - target_velocity

    block_edges = [
        min(plan.duration_s, index * block_ms / 1000.0)
        for index in range(len(plan.blocks) + 1)
    ]
    boundary_indices = np.clip(
        np.rint(np.asarray(block_edges) / (plan.slice_us / 1_000_000.0)).astype(int),
        1,
        len(output_velocity) - 1,
    )
    boundary_velocity_jump = np.abs(
        output_velocity[boundary_indices] - output_velocity[boundary_indices - 1]
    )

    return {
        "configuration": {
            "axis": axis.upper(), "amplitude_mm": amplitude_mm,
            "requested_period_s": period_s, "effective_period_s": effective_period,
            "cycles": cycles, "steps_per_mm": steps_per_mm,
            "slice_ms": slice_ms, "block_ms": block_ms,
            "isr_tick_us": ISR_TICK_US, "blocks": len(plan.blocks),
        },
        "metrics": {
            "max_position_error_um": float(np.max(np.abs(error_um))),
            "rms_position_error_um": float(np.sqrt(np.mean(error_um**2))),
            "max_velocity_error_mm_s": float(np.max(np.abs(velocity_error))),
            "rms_velocity_error_mm_s": float(np.sqrt(np.mean(velocity_error**2))),
            "peak_pulse_rate_hz": float(np.max(np.abs(plan.pulse_rate[:, axis_index]))),
            "max_boundary_velocity_change_mm_s": float(np.max(boundary_velocity_jump)),
            "zero_motion_slices": int(np.count_nonzero(plan.delta_steps[:, axis_index] == 0)),
            "final_step_error": int(plan.delta_steps[:, axis_index].sum()),
        },
        "position": [
            [round(float(a), 4), round(float(b), 5), round(float(c), 5), round(float(d), 3)]
            for a, b, c, d in zip(t, target, output, error_um)
        ],
        "velocity": [
            [round(float(a), 4), round(float(b), 5), round(float(c), 5), round(float(d), 1)]
            for a, b, c, d in zip(
                slice_center,
                target_velocity,
                output_velocity,
                plan.pulse_rate[:, axis_index],
            )
        ],
        "block_edges": [round(float(value), 4) for value in block_edges],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--axis", choices=("X", "Y"), default="X")
    parser.add_argument("--amplitude", type=float, default=20.0)
    parser.add_argument("--period", type=float, default=4.0)
    parser.add_argument("--cycles", type=float, default=2.0)
    args = parser.parse_args()
    result = simulate_sine(args.axis, args.amplitude, args.period, args.cycles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"configuration": result["configuration"], "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
