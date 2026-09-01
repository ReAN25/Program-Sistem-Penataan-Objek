"""Eksekusi trajektori GUI penuh menggunakan Pulse Block Streaming."""

from __future__ import annotations

import csv
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np


class PulseBlockExecutionMixin:
    # Sama dengan Segment Planner: titik gerak dipadatkan setiap 0,3 mm lalu
    # dihaluskan sebelum masuk ke planner waktu Pulse Block.
    RUN_TRAJECTORY_SPACING_MM = 0.3

    def run_sequence(self):
        if len(self.optimized_trajectory) == 0 and len(getattr(self, "main_waypoints", [])) > 0:
            self.log_msg("Belum ada hasil generate; membuat trajektori langsung...")
            self.generate_direct_trajectory()
        if len(self.optimized_trajectory) == 0:
            self.log_msg("Tidak ada trajektori untuk dijalankan.")
            return
        if self.is_running:
            self.log_msg("RUN sebelumnya masih aktif.")
            return
        self.stop_requested = False
        self.is_running = True
        threading.Thread(target=self._execute_pulse_run, daemon=True).start()

    # ------------------------------------------------------------------
    # Format log RUN mengikuti PROGRAM FULL SEGMENT PLANNER. Log disiapkan
    # sebelum blok pertama dikirim, sehingga STOP/error di tengah tetap
    # menghasilkan CSV dengan baris terakhir berstatus queued_or_interrupted.
    # ------------------------------------------------------------------
    def _start_run_log(self, cycle_mode):
        self._run_log_started_at = datetime.now()
        self._run_log_started_perf = time.perf_counter()
        self._run_log_cycle_mode = bool(cycle_mode)
        self._run_log_rows = []
        self._run_log_segment_index = 0
        self._run_log_saved = False

    @staticmethod
    def _run_log_number(value):
        if value == "":
            return ""
        try:
            return f"{float(value):.6f}"
        except (TypeError, ValueError):
            return value

    def _append_run_log_segment(
        self,
        mode,
        start_point,
        end_point,
        pass_label="",
        action_label="",
        distance_mm=None,
        arc_center=None,
        arc_sweep_deg=None,
    ):
        rows = getattr(self, "_run_log_rows", None)
        if rows is None:
            return None

        start = np.asarray(start_point, dtype=float)
        end = np.asarray(end_point, dtype=float)
        delta = end - start
        if distance_mm is None:
            distance_mm = float(np.linalg.norm(delta))
        else:
            distance_mm = float(distance_mm)
        if distance_mm <= 0.05:
            return None

        self._run_log_segment_index += 1
        center_x = ""
        center_y = ""
        if arc_center is not None:
            center = np.asarray(arc_center, dtype=float)
            center_x = float(center[0])
            center_y = float(center[1])

        row = {
            "segment_index": self._run_log_segment_index,
            "queued_at": datetime.now().isoformat(timespec="milliseconds"),
            "done_at": "",
            "segment_status": "queued_or_interrupted",
            "pass_label": pass_label,
            "mode": mode,
            "action_label": action_label,
            "start_x_mm": float(start[0]),
            "start_y_mm": float(start[1]),
            "start_z_mm": float(start[2]),
            "end_x_mm": float(end[0]),
            "end_y_mm": float(end[1]),
            "end_z_mm": float(end[2]),
            "dx_mm": float(delta[0]),
            "dy_mm": float(delta[1]),
            "dz_mm": float(delta[2]),
            "distance_mm": distance_mm,
            "arc_center_x_mm": center_x,
            "arc_center_y_mm": center_y,
            "arc_sweep_deg": "" if arc_sweep_deg is None else float(arc_sweep_deg),
        }
        rows.append(row)
        return row

    @staticmethod
    def _mark_run_log_done(row):
        if row is not None:
            row["segment_status"] = "done"
            row["done_at"] = datetime.now().isoformat(timespec="milliseconds")

    def _save_run_log(self, run_completed=False):
        if getattr(self, "_run_log_saved", False):
            return getattr(self, "_run_log_path", None)
        started_at = getattr(self, "_run_log_started_at", None)
        if started_at is None:
            return None

        log_dir = Path(getattr(self, "run_log_dir", Path(__file__).resolve().parents[1] / "run_logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        finished_at = datetime.now()
        total_time_s = max(
            0.0,
            time.perf_counter() - float(getattr(self, "_run_log_started_perf", time.perf_counter())),
        )
        rows = list(getattr(self, "_run_log_rows", []))
        total_distance_mm = sum(float(row["distance_mm"]) for row in rows)
        if run_completed:
            run_status = "completed"
        elif bool(getattr(self, "stop_requested", False)):
            run_status = "stopped_partial"
        else:
            run_status = "error_partial"

        cycle_label = "cycle_on" if bool(getattr(self, "_run_log_cycle_mode", False)) else "cycle_off"
        csv_path = log_dir / (
            f"run_{cycle_label}_{run_status}_"
            f"{started_at.strftime('%Y%m%d_%H%M%S_%f')}.csv"
        )
        fieldnames = [
            "run_started_at", "run_finished_at", "run_status", "cycle_mode",
            "total_time_s", "total_distance_mm", "segment_index", "queued_at",
            "done_at", "segment_status", "pass_label", "mode", "action_label",
            "start_x_mm", "start_y_mm", "start_z_mm", "end_x_mm", "end_y_mm",
            "end_z_mm", "dx_mm", "dy_mm", "dz_mm", "distance_mm",
            "arc_center_x_mm", "arc_center_y_mm", "arc_sweep_deg",
        ]
        if not rows:
            rows = [{
                "segment_index": 0, "queued_at": "", "done_at": "",
                "segment_status": "no_segment_sent", "pass_label": "", "mode": "",
                "action_label": "", "start_x_mm": "", "start_y_mm": "",
                "start_z_mm": "", "end_x_mm": "", "end_y_mm": "", "end_z_mm": "",
                "dx_mm": "", "dy_mm": "", "dz_mm": "", "distance_mm": 0.0,
                "arc_center_x_mm": "", "arc_center_y_mm": "", "arc_sweep_deg": "",
            }]

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                output = {
                    "run_started_at": started_at.isoformat(timespec="milliseconds"),
                    "run_finished_at": finished_at.isoformat(timespec="milliseconds"),
                    "run_status": run_status,
                    "cycle_mode": "on" if bool(getattr(self, "_run_log_cycle_mode", False)) else "off",
                    "total_time_s": self._run_log_number(total_time_s),
                    "total_distance_mm": self._run_log_number(total_distance_mm),
                }
                output.update(row)
                for key in (
                    "start_x_mm", "start_y_mm", "start_z_mm", "end_x_mm", "end_y_mm",
                    "end_z_mm", "dx_mm", "dy_mm", "dz_mm", "distance_mm",
                    "arc_center_x_mm", "arc_center_y_mm", "arc_sweep_deg",
                ):
                    output[key] = self._run_log_number(output[key])
                writer.writerow(output)

        self._run_log_saved = True
        self._run_log_path = csv_path
        return csv_path

    @staticmethod
    def _append_route_point(route, point):
        point = np.asarray(point, dtype=float)
        if not route or np.linalg.norm(route[-1] - point) > 1e-6:
            route.append(point.copy())
        return len(route) - 1

    def _build_rounded_trajectory(self, source_trajectory, preserve_action_points=False):
        """Bentuk titik RUN sama seperti Segment Planner.

        Titik kontrol/hasil FHO dipadatkan pada jarak tetap, dihaluskan dengan
        moving average berujung-blend, lalu dikuantisasi 0,1 mm. Pulse Block
        hanya mengganti tahap eksekusinya menjadi timeline pulsa; geometri rute
        tetap sama.

        Ketika ``preserve_action_points`` aktif, lintasan dipecah pada setiap
        waypoint GRIP/RELEASE sebelum moving average dilakukan. Dengan begitu
        smoothing tidak menggeser waypoint aksi internal (misalnya GRIP pertama
        pada lintasan panjang), sehingga checkpoint tetap dapat menghentikan
        stream untuk gerak offset dan servo.
        """
        source = np.asarray(source_trajectory, dtype=float)
        if len(source) == 0:
            return []
        if len(source) == 1:
            point = np.round(source[0], 1)
            point[2] = max(0.0, point[2])
            return [point]

        point_spacing = max(
            0.05,
            float(getattr(self, "run_trajectory_spacing_mm", self.RUN_TRAJECTORY_SPACING_MM)),
        )

        if preserve_action_points:
            waypoint_actions = getattr(self, "main_waypoint_actions", [])
            action_waypoints = [
                np.asarray(point, dtype=float).copy()
                for point, action in zip(
                    getattr(self, "main_waypoints", []), waypoint_actions
                )
                if action in ("GRIP", "RELEASE")
            ]
            if action_waypoints:
                # Sisipkan waypoint aksi ke sumber secara monoton. Ini tidak
                # mengubah jalur selain memastikan koordinat aksi benar-benar
                # menjadi titik data yang tidak hilang akibat smoothing.
                augmented = [source[0].copy()]
                action_indices = []
                source_index = 0
                for action_point in action_waypoints:
                    if source_index >= len(source):
                        break
                    remaining = source[source_index:]
                    nearest = source_index + int(
                        np.argmin(
                            np.linalg.norm(remaining - action_point[None, :], axis=1)
                        )
                    )
                    while source_index < nearest:
                        source_index += 1
                        augmented.append(source[source_index].copy())
                    rounded_action = np.round(action_point, 1)
                    if not np.array_equal(augmented[-1], rounded_action):
                        augmented.append(rounded_action)
                    action_indices.append(len(augmented) - 1)
                    source_index = nearest

                while source_index < len(source) - 1:
                    source_index += 1
                    augmented.append(source[source_index].copy())

                augmented_source = np.asarray(augmented, dtype=float)
                pieces = []
                boundaries = [0] + action_indices + [len(augmented_source) - 1]
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    if end < start:
                        continue
                    piece = self._build_rounded_trajectory(
                        augmented_source[start : end + 1],
                        preserve_action_points=False,
                    )
                    if not piece:
                        continue
                    # Moving average hanya boleh mengubah bagian dalam; kedua
                    # ujung segmen harus tetap persis pada waypoint sumber.
                    piece[0] = np.round(augmented_source[start], 1)
                    piece[-1] = np.round(augmented_source[end], 1)
                    for point in piece:
                        if not pieces or not np.array_equal(pieces[-1], point):
                            pieces.append(np.asarray(point, dtype=float).copy())
                if pieces:
                    return pieces

        smooth_trajectory = [source[0].copy()]
        for i in range(1, len(source)):
            p1 = np.asarray(smooth_trajectory[-1], dtype=float)
            p2 = source[i]
            dist = float(np.linalg.norm(p2 - p1))
            while dist > point_spacing:
                direction = (p2 - p1) / dist
                new_point = p1 + direction * point_spacing
                smooth_trajectory.append(new_point)
                p1 = new_point
                dist = float(np.linalg.norm(p2 - p1))
            smooth_trajectory.append(p2.copy())

        arr = np.asarray(smooth_trajectory, dtype=float)
        window = min(100, len(arr) // 4)
        if window >= 5:
            smoothed = arr.copy()
            for axis in range(3):
                pad_start = np.repeat(arr[0:1, axis], window)
                pad_end = np.repeat(arr[-1:, axis], window)
                padded = np.concatenate((pad_start, arr[:, axis], pad_end))
                cumsum = np.cumsum(np.insert(padded, 0, 0.0))
                moving_average = (cumsum[window:] - cumsum[:-window]) / float(window)
                start_idx = (len(moving_average) - len(arr)) // 2
                smoothed_axis = moving_average[start_idx : start_idx + len(arr)].copy()
                blend = np.linspace(0.0, 1.0, window)
                smoothed_axis[:window] = (
                    arr[:window, axis] * (1.0 - blend)
                    + smoothed_axis[:window] * blend
                )
                smoothed_axis[-window:] = (
                    arr[-window:, axis] * blend
                    + smoothed_axis[-window:] * (1.0 - blend)
                )
                smoothed[:, axis] = smoothed_axis
            smooth_trajectory = smoothed

        rounded_trajectory = []
        for point in smooth_trajectory:
            rounded = np.round(np.asarray(point, dtype=float), 1)
            rounded[2] = max(0.0, rounded[2])
            if not rounded_trajectory or not np.array_equal(rounded, rounded_trajectory[-1]):
                rounded_trajectory.append(rounded)
        if rounded_trajectory:
            rounded_trajectory[0] = np.round(source[0], 1)
            if len(rounded_trajectory) > 1:
                rounded_trajectory[-1] = np.round(source[-1], 1)
        return rounded_trajectory

    def _prepare_action_trajectory(self, trajectory):
        """Sisipkan waypoint aksi ke titik terdekat seperti Segment Planner."""
        waypoint_actions = getattr(self, "main_waypoint_actions", [])
        if len(trajectory) == 0 or len(waypoint_actions) == 0:
            return list(trajectory), {}

        trajectory_array = np.asarray(trajectory, dtype=float)
        prepared = []
        action_index_map = {}
        search_start = 0
        copy_start = 0
        waypoints = getattr(self, "main_waypoints", [])
        for waypoint, action in zip(waypoints, waypoint_actions):
            if action not in ("GRIP", "RELEASE"):
                continue
            if search_start >= len(trajectory_array):
                search_start = len(trajectory_array) - 1
            remaining = trajectory_array[search_start:]
            nearest = search_start + int(
                np.argmin(np.linalg.norm(remaining - np.asarray(waypoint, dtype=float), axis=1))
            )
            while copy_start <= nearest and copy_start < len(trajectory):
                prepared.append(np.round(np.asarray(trajectory[copy_start], dtype=float), 1))
                copy_start += 1
            action_point = np.round(np.asarray(waypoint, dtype=float), 1)
            if not prepared or not np.array_equal(prepared[-1], action_point):
                prepared.append(action_point)
            action_index_map.setdefault(len(prepared) - 1, []).append(action)
            search_start = min(nearest + 1, len(trajectory_array) - 1)

        while copy_start < len(trajectory):
            prepared.append(np.round(np.asarray(trajectory[copy_start], dtype=float), 1))
            copy_start += 1
        return prepared, action_index_map

    def _build_grip_release_route(
        self,
        prepared_trajectory,
        action_index_map,
        include_prefix=True,
        include_first_grip_action=True,
        include_last_return=False,
    ):
        action_events = [
            (index, action)
            for index in sorted(action_index_map)
            for action in action_index_map[index]
            if action in ("GRIP", "RELEASE")
        ]
        pairs = []
        active_grip_index = None
        for index, action in action_events:
            if action == "GRIP":
                active_grip_index = index
            elif action == "RELEASE" and active_grip_index is not None and index > active_grip_index:
                pairs.append((active_grip_index, index))
                active_grip_index = None
        if not pairs:
            return list(prepared_trajectory), dict(action_index_map), False

        route = []
        route_action_map = {}

        def append_point(point):
            next_point = np.round(np.asarray(point, dtype=float), 1)
            if not route or not np.array_equal(route[-1], next_point):
                route.append(next_point)
            return len(route) - 1

        def append_segment(segment):
            for point in segment:
                append_point(point)

        def add_action(action):
            if route:
                route_action_map.setdefault(len(route) - 1, []).append(action)

        first_grip_index = pairs[0][0]
        if include_prefix:
            append_segment(prepared_trajectory[: first_grip_index + 1])
            if include_first_grip_action:
                add_action("GRIP")

        for pair_index, (grip_index, release_index) in enumerate(pairs):
            outbound = [
                np.round(np.asarray(point, dtype=float), 1)
                for point in prepared_trajectory[grip_index : release_index + 1]
            ]
            if len(outbound) < 2:
                continue
            if not route:
                append_point(outbound[0])
                if include_first_grip_action:
                    add_action("GRIP")
            append_segment(outbound[1:])
            add_action("RELEASE")
            if include_last_return or pair_index < len(pairs) - 1:
                append_segment(outbound[-2::-1])
                add_action("GRIP")
        return route, route_action_map, True

    def _route_from_generated_segments(self, include_last_return=False):
        segments = list(getattr(self, "optimized_grip_release_segments", []))
        if not segments:
            return None
        route, actions = [], {}
        prefix = np.asarray(getattr(self, "optimized_prefix_trajectory", []), dtype=float)
        if len(prefix):
            for point in self._build_rounded_trajectory(prefix):
                self._append_route_point(route, point)
        elif self.grip_point is not None:
            for point in self._build_rounded_trajectory([
                np.zeros(3), np.asarray(self.grip_point, dtype=float)
            ]):
                self._append_route_point(route, point)
        if route:
            actions.setdefault(len(route) - 1, []).append("GRIP")

        for index, entry in enumerate(segments):
            outbound = self._build_rounded_trajectory(self._get_segment_path(entry, "outbound"))
            returning = self._build_rounded_trajectory(self._get_segment_path(entry, "return"))
            for point in outbound:
                self._append_route_point(route, point)
            actions.setdefault(len(route) - 1, []).append("RELEASE")
            if include_last_return or index < len(segments) - 1:
                return_points = returning if len(returning) >= 2 else outbound[-2::-1]
                for point in return_points:
                    self._append_route_point(route, point)
                actions.setdefault(len(route) - 1, []).append("GRIP")
        return route, actions

    def _route_from_trajectory(self, include_last_return=False):
        rounded = self._build_rounded_trajectory(
            self.optimized_trajectory,
            preserve_action_points=True,
        )
        prepared, action_map = self._prepare_action_trajectory(rounded)
        if bool(getattr(self, "direct_sequence_mode", False)):
            return prepared, action_map
        return self._build_grip_release_route(
            prepared,
            action_map,
            include_prefix=True,
            include_first_grip_action=True,
            include_last_return=bool(include_last_return),
        )[:2]

    def _build_run_route(self, include_last_return=False):
        generated = self._route_from_generated_segments(include_last_return)
        if generated is not None:
            return generated
        return self._route_from_trajectory(include_last_return)

    def _wait_servo_ack(self, action):
        if not self.is_arduino_connected():
            # Samakan simulasi offline dengan waktu mekanis GRIP di firmware.
            # Pada koneksi nyata _send_grip_command menunggu GOK dari Arduino.
            if action == "GRIP":
                time.sleep(float(getattr(self, "SERVO_GRIP_ACTIVE_S", 1.0)))
            else:
                time.sleep(0.15)
            return True
        # Fungsi pengiriman sudah menunggu ACK dan memeriksa STATE sebelum
        # mengulang, sehingga aksi servo bersifat idempoten.
        return self._send_grip_command(action)

    def _execute_action(self, action, trajectory_point, pass_label=""):
        # Ambil nilai terbaru dari GUI ketika RUN dimulai. Pada object uji atau
        # mode offline helper ini tidak wajib tersedia, sehingga nilai cache
        # tetap dipakai sebagai fallback.
        refresh_offset = getattr(self, "_grip_release_z_offset", None)
        if callable(refresh_offset):
            offset = float(refresh_offset())
        else:
            offset = float(getattr(self, "grip_release_z_offset", 0.0))
        action_point = self._action_point_from_trajectory_point(trajectory_point)
        self.log_msg(
            f"Aksi {action}: trajektori Z={float(trajectory_point[2]):.1f} mm, "
            f"offset +{offset:.1f} mm -> posisi aksi Z={action_point[2]:.1f} mm"
        )
        down_row = self._append_run_log_segment(
            "ACTION_GOTO",
            np.asarray(self._hardware_pos, dtype=float),
            action_point,
            pass_label=pass_label,
            action_label="DOWN",
        )
        self._stream_points_blocking([self._hardware_pos, action_point])
        self._mark_run_log_done(down_row)
        self.current_pos = action_point.copy()
        self.root.after(0, self.update_plot)
        if not self._wait_servo_ack(action):
            return False

        if action == "GRIP":
            self._place_bottle_obstacle(action_point)
            self._apply_grip_state(True)
            self.bottle_bottom = action_point.copy()
        else:
            self._place_bottle_obstacle(action_point)
            self._apply_grip_state(False)
        self.root.after(0, self.update_plot)

        if not self.is_running:
            return False
        up_row = self._append_run_log_segment(
            "ACTION_GOTO",
            np.asarray(self._hardware_pos, dtype=float),
            trajectory_point,
            pass_label=pass_label,
            action_label="UP",
        )
        self._stream_points_blocking([self._hardware_pos, trajectory_point])
        self._mark_run_log_done(up_row)
        self.current_pos = np.asarray(trajectory_point, dtype=float).copy()
        self.root.after(0, self.update_plot)
        return self.is_running

    def _stream_route_once(self, route, actions, label="awal"):
        if not route:
            return False
        start_index = 0
        action_indices = sorted(index for index in actions if 0 <= index < len(route))
        checkpoints = action_indices + ([len(route) - 1] if len(route) - 1 not in action_indices else [])
        for end_index in checkpoints:
            if not self.is_running:
                return False
            chunk = route[start_index:end_index + 1]
            if not chunk:
                chunk = [route[end_index]]
            self.log_msg(
                f"Pulse Block {label}: titik {start_index}..{end_index}, "
                f"tanpa berhenti pada waypoint antara."
            )
            segment_row = self._append_run_log_segment(
                "LINE",
                np.asarray(self._hardware_pos, dtype=float),
                np.asarray(route[end_index], dtype=float),
                pass_label=label,
            )
            plan = self._stream_points_blocking(chunk, progress=None)
            self._mark_run_log_done(segment_row)
            self.current_pos = np.asarray(route[end_index], dtype=float).copy()
            self.root.after(0, self.update_plot)
            if plan is not None:
                self.log_msg(
                    f"Selesai {len(plan.blocks)} blok / {len(plan.delta_steps)} slice / "
                    f"{plan.duration_s:.2f} s terencana."
                )
            checkpoint_actions = actions.get(end_index, [])
            if checkpoint_actions:
                self.log_msg(
                    f"Checkpoint aksi tercapai di X:{route[end_index][0]:.1f} "
                    f"Y:{route[end_index][1]:.1f} Z:{route[end_index][2]:.1f}; "
                    "stream dihentikan sementara untuk offset dan servo."
                )
            for action in checkpoint_actions:
                if not self._execute_action(action, route[end_index], pass_label=label):
                    return False
            start_index = end_index
        return self.is_running

    def _execute_pulse_run(self):
        self._start_run_log(self.cycle_mode)
        run_completed = False
        self._hardware_pos = np.asarray(
            getattr(self, "_hardware_pos", np.zeros(3)), dtype=float
        ).copy()
        self.current_pos = self._hardware_pos.copy()
        self.is_gripping = False
        self.bottle_bottom = None
        self.root.after(0, self._update_grip_button)
        self.root.after(0, self._set_end_effector_visuals_visible, True)
        try:
            refresh_offset = getattr(self, "_grip_release_z_offset", None)
            if callable(refresh_offset):
                refresh_offset()
            if self.is_arduino_connected():
                with self.serial_command_lock:
                    if not self._send_current_config_to_arduino():
                        raise RuntimeError("Konfigurasi aktuator gagal dikirim ke Arduino.")
                    _units, actual_position, state = self._read_hardware_position()
                if int(state.get("HOME", "0")) != 1:
                    raise RuntimeError(
                        "Arduino belum homing (HOME=0). Tekan HOME dan tunggu HOMING_END "
                        "sebelum RUN trajektori."
                    )
                if int(state.get("ZENC", "0")) != 1:
                    raise RuntimeError("Encoder Z tidak siap (ZENC=0); RUN dibatalkan.")
                self._hardware_pos = actual_position.copy()
                self.current_pos = actual_position.copy()
            route, actions = self._build_run_route(include_last_return=self.cycle_mode)
            self.log_msg(
                f"RUN Pulse Block XYZ-angle: {len(route)} titik geometris, {len(actions)} checkpoint aksi, "
                f"slice {self.pulse_slice_ms} ms, blok {self.pulse_block_ms} ms, buffer 32 slot."
            )
            self.log_msg(
                "Profil waktu XYZ disinkronkan dengan plot; target Z tidak "
                "dikirim saat koordinat Z pada trajektori tidak berubah."
            )
            self.log_msg(
                "Kecepatan setiap sumbu konstan pada seluruh slice; "
                "tidak ada percepatan atau perlambatan. "
                "Robot hanya berhenti pada checkpoint Grip/Release."
            )
            if not self._stream_route_once(route, actions, "awal"):
                return
            loop_count = 1
            while self.is_running and self.cycle_mode:
                loop_count += 1
                route, actions = self._build_run_route(include_last_return=True)
                if not self._stream_route_once(route, actions, f"loop {loop_count}"):
                    return
            if self.is_running:
                run_completed = True
                self.log_msg("Eksekusi Pulse Block selesai tanpa penghentian antar-waypoint.")
        except Exception as exc:
            self.log_msg(f"RUN Pulse Block gagal: {exc}")
            if self.is_arduino_connected():
                try:
                    _units, position, values = self._read_hardware_position()
                    self.current_pos = position.copy()
                    stream_busy = any(
                        int(values.get(key, "0"))
                        for key in ("RUN", "XYRUN", "XYWAIT")
                    )
                    if stream_busy and not self.stop_requested:
                        self.log_msg(
                            "STATE menunjukkan stream lama masih aktif; "
                            "STOP otomatis dikirim sebelum kontrol berikutnya."
                        )
                        self._stop_stream_and_wait("RUN gagal")
                        try:
                            _units, position, _values = self._read_hardware_position()
                            self.current_pos = position.copy()
                        except Exception as state_after_stop:
                            self.log_msg(
                                f"STATE setelah STOP belum tersedia: {state_after_stop}"
                            )
                    self.root.after(0, self.update_plot)
                except Exception as state_exc:
                    self.log_msg(f"STATE pemulihan RUN tidak tersedia: {state_exc}")
        finally:
            try:
                saved_run_path = self._save_run_log(run_completed)
                if saved_run_path is not None:
                    self.log_msg(
                        f"CSV RUN tersimpan: {saved_run_path} "
                        f"({len(getattr(self, '_run_log_rows', []))} segmen; "
                        f"status={'completed' if run_completed else 'partial'})."
                    )
            except Exception as log_exc:
                self.log_msg(f"Gagal menyimpan CSV RUN: {log_exc}")
            self.is_running = False
