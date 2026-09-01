"""Perencana trajektori berbasis waktu untuk firmware Pulse Block Streaming."""

from __future__ import annotations

import binascii
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


# Timer firmware 20 us menghasilkan pulsa HIGH 2 us di dalam satu ISR,
# sehingga kapasitas teoritis tetap 50.000 pulsa/detik.
ISR_TICK_US = 20
MAX_SLICES_PER_BLOCK = 2
MOTOR_STEPS_PER_REV = 6400.0
Z_MOTOR_TO_OUTPUT_REDUCTION = 20.0
Z_MOTOR_STEPS_PER_OUTPUT_DEG = (
    MOTOR_STEPS_PER_REV * Z_MOTOR_TO_OUTPUT_REDUCTION / 360.0
)


@dataclass(frozen=True)
class PulseBlock:
    sequence: int
    slices: tuple[tuple[int, int, int], ...]

    @property
    def payload(self) -> str:
        return ";".join(f"{dx},{dy},{z_target}" for dx, dy, z_target in self.slices)

    @property
    def crc16(self) -> int:
        return binascii.crc_hqx(self.payload.encode("ascii"), 0xFFFF)

    @property
    def command(self) -> str:
        return (
            f"BLOCK:{self.sequence}:{len(self.slices)}:"
            f"{self.crc16:04X}:{self.payload}\n"
        )


@dataclass
class PulsePlan:
    times_s: np.ndarray
    desired_mm: np.ndarray
    planned_mm: np.ndarray
    delta_steps: np.ndarray
    steps_per_mm: np.ndarray
    z_deg_per_cm: float
    slice_us: int
    block_ms: int
    # Transport boleh melepas blok yang telah selesai; sequence tetap dapat
    # direkonstruksi dari array delta bila firmware meminta retry.
    blocks: list[PulseBlock | None]

    @property
    def duration_s(self) -> float:
        return float(self.times_s[-1]) if len(self.times_s) else 0.0

    @property
    def pulse_rate(self) -> np.ndarray:
        return self.delta_steps.astype(float) / (self.slice_us / 1_000_000.0)

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    def get_block(self, sequence: int) -> PulseBlock:
        """Ambil blok berdasarkan sequence, membangun ulang bila sudah dilepas."""
        sequence = int(sequence)
        if sequence < 0 or sequence >= len(self.blocks):
            raise IndexError(f"Sequence blok di luar plan: {sequence}")
        block = self.blocks[sequence]
        if block is not None:
            return block

        slices_per_block = max(1, self.block_ms * 1000 // self.slice_us)
        start = sequence * slices_per_block
        stop = min(len(self.delta_steps), start + slices_per_block)
        if start >= stop:
            raise IndexError(f"Data slice untuk sequence {sequence} tidak tersedia")
        z_targets = np.rint(
            self.desired_mm[start + 1 : stop + 1, 2] * self.z_deg_per_cm
        ).astype(np.int64)
        slices = tuple(
            (int(row[0]), int(row[1]), int(z_target))
            for row, z_target in zip(self.delta_steps[start:stop], z_targets)
        )
        block = PulseBlock(sequence, slices)
        self.blocks[sequence] = block
        return block

    def release_block(self, sequence: int) -> None:
        """Lepaskan objek blok yang sudah selesai agar plan panjang hemat RAM."""
        sequence = int(sequence)
        if 0 <= sequence < len(self.blocks):
            self.blocks[sequence] = None


def _validate_timing(slice_ms: int, block_ms: int) -> tuple[int, int]:
    slice_ms = int(slice_ms)
    block_ms = int(block_ms)
    if slice_ms < 2 or slice_ms > 50:
        raise ValueError("Slice harus berada antara 2 dan 50 ms.")
    if block_ms < slice_ms:
        raise ValueError("Durasi blok harus lebih besar atau sama dengan slice.")
    slices_per_block = block_ms // slice_ms
    if slices_per_block < 1 or slices_per_block > MAX_SLICES_PER_BLOCK:
        raise ValueError(
            f"Jumlah slice per blok harus 1..{MAX_SLICES_PER_BLOCK}; "
            "ubah durasi blok atau slice."
        )
    return slice_ms * 1000, slices_per_block


def _make_plan(
    times_s: np.ndarray,
    desired_mm: np.ndarray,
    steps_per_cm: Iterable[float],
    z_deg_per_cm: float,
    slice_ms: int,
    block_ms: int,
) -> PulsePlan:
    slice_us, slices_per_block = _validate_timing(slice_ms, block_ms)
    steps_per_mm = np.asarray(tuple(steps_per_cm), dtype=float) / 10.0
    if steps_per_mm.shape != (2,) or np.any(steps_per_mm <= 0):
        raise ValueError("Kalibrasi X dan Y harus lebih besar dari nol.")
    if desired_mm.ndim != 2 or desired_mm.shape[1] != 3:
        raise ValueError("Posisi harus berbentuk array N x 3.")
    if len(times_s) != len(desired_mm) or len(times_s) < 2:
        raise ValueError("Data waktu dan posisi tidak lengkap.")
    if z_deg_per_cm <= 0:
        raise ValueError("Kalibrasi Z derajat/cm harus lebih besar dari nol.")

    target_steps = np.rint(desired_mm[:, :2] * steps_per_mm).astype(np.int64)
    delta_steps = np.diff(target_steps, axis=0)
    # Target Z adalah sudut poros OUTPUT gearbox dalam deci-degree.
    # Tidak ada faktor reduksi di sini: encoder memang berada di output.
    z_target_deci_deg = np.rint(desired_mm[:, 2] * z_deg_per_cm).astype(np.int64)
    if np.any(z_target_deci_deg < -32768) or np.any(z_target_deci_deg > 32767):
        raise ValueError("Target sudut Z melampaui format int16 deci-degree.")
    ticks_per_slice = slice_us // ISR_TICK_US
    max_steps_per_slice = ticks_per_slice
    largest_xy = int(np.max(np.abs(delta_steps))) if delta_steps.size else 0
    if largest_xy > max_steps_per_slice:
        raise ValueError(
            f"Laju pulsa X/Y terlalu tinggi: {largest_xy} langkah/slice, batas firmware "
            f"{max_steps_per_slice}. Kurangi kecepatan mm/s atau gunakan tick ISR yang lebih cepat."
        )
    if np.any(np.abs(delta_steps) > 32767):
        raise ValueError("Perubahan langkah per slice melampaui int16.")

    planned_mm = np.asarray(desired_mm, dtype=float).copy()
    planned_mm[:, :2] = target_steps.astype(float) / steps_per_mm
    blocks: list[PulseBlock | None] = []
    for start in range(0, len(delta_steps), slices_per_block):
        raw = delta_steps[start : start + slices_per_block]
        z_raw = z_target_deci_deg[start + 1 : start + 1 + len(raw)]
        slices = tuple(
            (int(row[0]), int(row[1]), int(z_target))
            for row, z_target in zip(raw, z_raw)
        )
        blocks.append(PulseBlock(len(blocks), slices))

    return PulsePlan(
        times_s=np.asarray(times_s, dtype=float),
        desired_mm=np.asarray(desired_mm, dtype=float),
        planned_mm=planned_mm,
        delta_steps=delta_steps.astype(np.int32),
        steps_per_mm=steps_per_mm,
        z_deg_per_cm=float(z_deg_per_cm),
        slice_us=slice_us,
        block_ms=int(block_ms),
        blocks=blocks,
    )


def plan_sinusoidal(
    axis: str = "X",
    amplitude_mm: float = 20.0,
    period_s: float = 4.0,
    cycles: float = 2.0,
    steps_per_cm: Iterable[float] = (2800.0, 1400.0),
    z_deg_per_cm: float = 15.0,
    slice_ms: int = 10,
    block_ms: int = 20,
) -> PulsePlan:
    """Buat gerak sinusoidal relatif yang mulai dan selesai di posisi nol."""
    axis_index = {"X": 0, "Y": 1}.get(axis.upper())
    if axis_index is None:
        raise ValueError("Sumbu motor harus X atau Y.")
    if amplitude_mm <= 0 or period_s <= 0 or cycles <= 0:
        raise ValueError("Amplitudo, periode, dan siklus harus lebih besar dari nol.")

    requested_duration_s = period_s * cycles
    slice_s = slice_ms / 1000.0
    slice_count = int(np.ceil(requested_duration_s / slice_s))
    duration_s = slice_count * slice_s
    effective_period_s = duration_s / cycles
    times_s = np.arange(slice_count + 1, dtype=float) * slice_s
    positions = np.zeros((len(times_s), 3), dtype=float)
    positions[:, axis_index] = amplitude_mm * np.sin(
        2.0 * np.pi * times_s / effective_period_s
    )
    positions[-1, axis_index] = 0.0
    return _make_plan(times_s, positions, steps_per_cm, z_deg_per_cm, slice_ms, block_ms)


def load_trajectory_csv(path: str | Path) -> np.ndarray:
    """Membaca CSV sederhana maupun bundle Segment Planner versi 2."""
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV tidak memiliki header.")
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        required = {"x_mm", "y_mm", "z_mm"}
        if not required.issubset(fields):
            raise ValueError("CSV harus memiliki kolom x_mm, y_mm, dan z_mm.")

        points = []
        for row in reader:
            record_type = row.get(fields.get("record_type", ""), "").strip().upper()
            if "record_type" in fields and record_type != "TRAJECTORY":
                continue
            try:
                points.append([
                    float(row[fields["x_mm"]]),
                    float(row[fields["y_mm"]]),
                    float(row[fields["z_mm"]]),
                ])
            except (TypeError, ValueError):
                continue
    if len(points) < 2:
        raise ValueError("CSV tidak memuat minimal dua titik trajektori.")
    result = np.asarray(points, dtype=float)
    if np.linalg.norm(result[0]) > 1e-9:
        result = np.vstack((np.zeros(3), result))
    return result


def plan_polyline(
    points_mm: np.ndarray,
    max_speed_mm_s: float | Iterable[float] = 80.0,
    steps_per_cm: Iterable[float] = (2800.0, 1400.0),
    z_deg_per_cm: float = 15.0,
    slice_ms: int = 10,
    block_ms: int = 20,
    constant_speed: bool | None = None,
) -> PulsePlan:
    """Parameterisasi waktu kontinu dengan kecepatan konstan pada setiap slice.

    Tidak ada penghentian pada titik CSV. Titik tetap dipakai sebagai geometri
    polyline; perubahan arah tajam tetap harus sudah aman dari proses planner.
    Semua sumbu menggunakan waktu yang sama agar target Z selalu sesuai dengan
    posisi pada plot. Parameter ``constant_speed`` dipertahankan hanya untuk
    kompatibilitas dengan pemanggil versi lama dan tidak lagi mengubah profil;
    seluruh pemanggilan selalu menggunakan timeline linear tanpa percepatan
    maupun perlambatan.
    """
    points = np.asarray(points_mm, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("Trajektori harus berisi minimal dua titik XYZ.")
    axis_speed_mm_s = np.asarray(max_speed_mm_s, dtype=float)
    if axis_speed_mm_s.ndim == 0:
        axis_speed_mm_s = np.repeat(axis_speed_mm_s, 3)
    elif axis_speed_mm_s.shape == (2,):
        axis_speed_mm_s = np.append(axis_speed_mm_s, axis_speed_mm_s[-1])
    if axis_speed_mm_s.shape != (3,) or np.any(axis_speed_mm_s <= 0):
        raise ValueError("Kecepatan X, Y, dan Z harus lebih besar dari nol.")

    delta_xyz = np.diff(points, axis=0)
    moving_xyz = np.linalg.norm(delta_xyz, axis=1)
    keep = np.concatenate(([True], moving_xyz > 1e-9))
    points = points[keep]
    if len(points) < 2:
        raise ValueError("Semua titik trajektori sama.")
    delta_xyz = np.diff(points, axis=0)
    steps_per_mm_xy = np.asarray(tuple(steps_per_cm), dtype=float) / 10.0
    if steps_per_mm_xy.shape != (2,) or np.any(steps_per_mm_xy <= 0):
        raise ValueError("Kalibrasi X dan Y harus lebih besar dari nol.")
    # Hanya perhitungan kebutuhan pulsa motor yang memakai reduksi 20:1.
    # Sudut target/feedback tetap sudut output encoder.
    z_steps_per_mm = (z_deg_per_cm / 10.0) * Z_MOTOR_STEPS_PER_OUTPUT_DEG
    timer_axis_capacity_mm_s = np.asarray([
        (1_000_000.0 / ISR_TICK_US) / steps_per_mm_xy[0],
        (1_000_000.0 / ISR_TICK_US) / steps_per_mm_xy[1],
        (1_000_000.0 / ISR_TICK_US / 2.0) / z_steps_per_mm,
    ])
    effective_axis_speed = np.minimum(axis_speed_mm_s, timer_axis_capacity_mm_s)
    # Samakan timeline dengan kecepatan diskrit interval timer Z pada firmware.
    timer_hz = 1_000_000.0 / ISR_TICK_US
    requested_z_pulse_hz = effective_axis_speed[2] * z_steps_per_mm
    z_interval_ticks = max(2, int(timer_hz / requested_z_pulse_hz + 0.5))
    actual_z_speed_mm_s = timer_hz / z_interval_ticks / z_steps_per_mm
    effective_axis_speed[2] = min(effective_axis_speed[2], actual_z_speed_mm_s)
    segment_base_time = np.max(np.abs(delta_xyz) / effective_axis_speed, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_base_time)))
    total_base_time = float(cumulative[-1])
    # Timeline selalu linear. Kecepatan tiap axis ditetapkan oleh
    # ``effective_axis_speed`` dan dipertahankan sepanjang slice; tidak ada
    # profil percepatan/perlambatan tersembunyi ketika CYCLE OFF maupun ON.
    requested_duration_s = max(0.1, total_base_time)
    slice_s = slice_ms / 1000.0
    slice_count = int(np.ceil(requested_duration_s / slice_s))
    duration_s = slice_count * slice_s
    times_s = np.arange(slice_count + 1, dtype=float) * slice_s
    u = times_s / duration_s
    motion_time = u * total_base_time

    positions = np.empty((len(times_s), 3), dtype=float)
    segment_index = np.searchsorted(cumulative, motion_time, side="right") - 1
    segment_index = np.clip(segment_index, 0, len(segment_base_time) - 1)
    local = (motion_time - cumulative[segment_index]) / segment_base_time[segment_index]
    positions[:] = points[segment_index] + local[:, None] * (
        points[segment_index + 1] - points[segment_index]
    )
    positions[0] = points[0]
    positions[-1] = points[-1]
    return _make_plan(times_s, positions, steps_per_cm, z_deg_per_cm, slice_ms, block_ms)
