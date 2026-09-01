import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.interpolate import splev, splprep


class OptimizationMixin:
    FHO_CONTROL_POINTS_PER_SEGMENT = 10
    FHO_POPULATION_SIZE = 20
    FHO_FIRE_HAWK_COUNT = 6
    FHO_MAX_EPOCH = 300
    FHO_MIN_SPLINE_SAMPLES = 400
    FHO_SPLINE_SAMPLE_MULTIPLIER = 25
    FHO_SHOW_TRAINING_CONTROL_POINTS = False
    FHO_SHOW_TRAINING_DYNAMIC_OBSTACLES = False
    FHO_BASELINE_RANDOM_RADIUS_MM = 10
    FHO_COLLISION_SAMPLE_MM = 10.0
    # Bobot ini hanya untuk kualitas geometri trajektori. Kecepatan, waktu,
    # percepatan, dan jerk tidak ikut dihitung di dalam FHO.
    FHO_LENGTH_WEIGHT = 0.70
    FHO_SMOOTHNESS_WEIGHT = 0.30
    FHO_SMOOTHNESS_SCALE_MM = 80.0
    FHO_SAFE_CLEARANCE_MM = 1.0
    FHO_CLEARANCE_WEIGHT = 1.50
    FHO_QUINTIC_SPLINE_DEGREE = 5
    TRAINING_LOG_FIELDS = [
        "training_started_at",
        "segment",
        "epoch",
        "role",
        "rank",
        "candidate_id",
        "fitness",
        "objective_j",
        "distance_mm",
        "valid",
        "control_point_index",
        "x_mm",
        "y_mm",
        "z_mm",
    ]

    @staticmethod
    def calculate_trajectory_distance_mm(points):
        """Hitung panjang total lintasan poliline XYZ dalam milimeter.

        Jarak dihitung sebagai penjumlahan jarak Euclidean setiap pasangan
        titik berurutan. Dengan demikian lintasan yang berbelok tidak
        direduksi menjadi jarak garis lurus antara titik awal dan akhir.
        """
        segment_lengths = OptimizationMixin.calculate_segment_distances_mm(points)
        return float(np.sum(segment_lengths))

    @staticmethod
    def calculate_segment_distances_mm(points):
        """Hitung jarak setiap segmen berurutan pada lintasan XYZ.

        Elemen ke-*i* pada hasil adalah jarak dari titik ``i`` ke titik
        ``i+1``. Titik awal homing tetap diperlakukan sebagai titik pertama
        apabila diberikan oleh pemanggil.
        """
        point_array = np.asarray(points, dtype=float)
        if point_array.size == 0:
            return np.array([], dtype=float)
        if point_array.ndim == 1:
            point_array = point_array.reshape(1, -1)
        if point_array.ndim != 2 or point_array.shape[1] < 3:
            raise ValueError("Trajektori harus berupa array N x 3 (XYZ).")

        xyz = point_array[:, :3]
        if not np.all(np.isfinite(xyz)):
            raise ValueError("Trajektori mengandung koordinat non-finite.")
        if len(xyz) < 2:
            return np.array([], dtype=float)

        return np.linalg.norm(np.diff(xyz, axis=0), axis=1)

    @staticmethod
    def calculate_axis_distances_mm(points):
        """Hitung total perpindahan absolut masing-masing sumbu X, Y, dan Z."""
        point_array = np.asarray(points, dtype=float)
        if point_array.size == 0:
            return np.zeros(3, dtype=float)
        if point_array.ndim == 1:
            point_array = point_array.reshape(1, -1)
        if point_array.ndim != 2 or point_array.shape[1] < 3:
            raise ValueError("Trajektori harus berupa array N x 3 (XYZ).")

        xyz = point_array[:, :3]
        if not np.all(np.isfinite(xyz)):
            raise ValueError("Trajektori mengandung koordinat non-finite.")
        if len(xyz) < 2:
            return np.zeros(3, dtype=float)

        return np.sum(np.abs(np.diff(xyz, axis=0)), axis=0)

    def _create_training_log_dir(self):
        log_root = getattr(self, "training_log_root", None)
        if log_root is None:
            log_root = Path(__file__).resolve().parents[1] / "data_training"

        log_root = Path(log_root)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_dir = log_root / f"training_{timestamp}"
        suffix = 1

        while training_dir.exists():
            training_dir = log_root / f"training_{timestamp}_{suffix:02d}"
            suffix += 1

        training_dir.mkdir(parents=True, exist_ok=True)
        self.current_training_started_at = timestamp
        self.current_training_log_dir = training_dir
        return training_dir

    def _training_segment_csv_path(self, segment_label):
        log_dir = getattr(self, "current_training_log_dir", None)
        if log_dir is None:
            return None

        clean_label = re.sub(r"[^0-9A-Za-z_-]+", "_", str(segment_label)).strip("_-")
        if not clean_label:
            clean_label = "segment"

        return Path(log_dir) / f"{clean_label}.csv"

    def _write_training_config(self, training_dir, collision_penalty):
        population_size = int(self.FHO_POPULATION_SIZE)
        fire_hawk_count = max(
            1,
            min(int(self.FHO_FIRE_HAWK_COUNT), population_size - 1),
        )
        config = {
            "training_started_at": self.current_training_started_at,
            "algorithm": "Fire Hawk Optimizer",
            "control_points_per_segment": int(self.FHO_CONTROL_POINTS_PER_SEGMENT),
            "population_size": population_size,
            "fire_hawk_count": fire_hawk_count,
            "prey_count": population_size - fire_hawk_count,
            "max_epoch": int(self.FHO_MAX_EPOCH),
            "exploration_radius_mm": float(self.FHO_BASELINE_RANDOM_RADIUS_MM),
            "collision_sample_mm": float(self.FHO_COLLISION_SAMPLE_MM),
            "min_spline_samples": int(self.FHO_MIN_SPLINE_SAMPLES),
            "spline_sample_multiplier": int(self.FHO_SPLINE_SAMPLE_MULTIPLIER),
            "collision_penalty": float(collision_penalty),
            "length_weight": float(self.FHO_LENGTH_WEIGHT),
            "smoothness_weight": float(self.FHO_SMOOTHNESS_WEIGHT),
            "smoothness_scale_mm": float(self.FHO_SMOOTHNESS_SCALE_MM),
            "safe_clearance_mm": float(self.FHO_SAFE_CLEARANCE_MM),
            "clearance_weight": float(self.FHO_CLEARANCE_WEIGHT),
            "spline_degree": int(self.FHO_QUINTIC_SPLINE_DEGREE),
            "baseline_mode": "polyline bebas collision; Z mengikuti detour",
            "initial_line_seed_mode": "control point tersebar sepanjang garis; Z=0",
            "gripper_size_mm": [float(value) for value in self.gripper_size],
            "bottle_size_mm": [float(value) for value in self.bottle_size],
            "workspace_mm": [
                float(self.workspace_x_max),
                float(self.workspace_y_max),
                float(self.workspace_z_max),
            ],
        }
        config_path = Path(training_dir) / "training_config.json"
        with config_path.open("w", encoding="utf-8") as config_file:
            json.dump(config, config_file, indent=2, ensure_ascii=False)
        return config_path

    def _training_float(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return ""

        if not np.isfinite(value):
            return ""

        return f"{value:.6f}"

    def _write_training_candidate_rows(
        self,
        writer,
        segment_label,
        epoch,
        role,
        rank,
        fitness,
        objective,
        distance,
        valid,
        points,
    ):
        if writer is None or points is None:
            return

        point_array = np.asarray(points, dtype=float)
        candidate_id = f"{role}_{int(rank):02d}"
        started_at = getattr(self, "current_training_started_at", "")

        for point_index, point in enumerate(point_array, start=1):
            writer.writerow({
                "training_started_at": started_at,
                "segment": segment_label,
                "epoch": int(epoch),
                "role": role,
                "rank": int(rank),
                "candidate_id": candidate_id,
                "fitness": self._training_float(fitness),
                "objective_j": self._training_float(objective),
                "distance_mm": self._training_float(distance),
                "valid": int(bool(valid)),
                "control_point_index": point_index,
                "x_mm": self._training_float(point[0]),
                "y_mm": self._training_float(point[1]),
                "z_mm": self._training_float(point[2]),
            })

    def _append_training_epoch_csv(
        self,
        segment_label,
        epoch,
        fitness,
        objectives,
        distances,
        valid_flags,
        repaired_points,
        global_best_points,
        global_best_fitness,
        global_best_objective,
        global_best_distance,
        fire_hawk_count,
    ):
        display_label = re.sub(r"[^0-9A-Za-z_-]+", "_", str(segment_label)).strip("_-")
        if not display_label:
            display_label = "segment"

        csv_path = self._training_segment_csv_path(display_label)
        if csv_path is None:
            return None

        try:
            file_exists = csv_path.exists()
            with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.TRAINING_LOG_FIELDS)
                if not file_exists:
                    writer.writeheader()

                for rank, points in enumerate(repaired_points, start=1):
                    role = "firehawk" if rank <= fire_hawk_count else "prey"
                    self._write_training_candidate_rows(
                        writer,
                        display_label,
                        epoch,
                        role,
                        rank,
                        fitness[rank - 1],
                        objectives[rank - 1],
                        distances[rank - 1],
                        valid_flags[rank - 1],
                        points,
                    )

                self._write_training_candidate_rows(
                    writer,
                    display_label,
                    epoch,
                    "global_best",
                    0,
                    global_best_fitness,
                    global_best_objective,
                    global_best_distance,
                    global_best_points is not None,
                    global_best_points,
                )
        except Exception as exc:
            self.current_training_log_dir = None
            try:
                self.log_msg(f"Gagal menulis CSV training {display_label}: {exc}")
            except Exception:
                pass
            return None

        return csv_path

    def _clear_grip_release_line_artists(self):
        self._remove_dynamic_artists(getattr(self, "grip_release_line_artists", []))
        self.grip_release_line_artists = []

    def _plot_optimized_trajectory(self):
        self._clear_grip_release_line_artists()

        if len(self.optimized_trajectory) == 0:
            self.opt_line.set_data([], [])
            self.opt_line.set_3d_properties([])
            return

        if bool(getattr(self, "direct_sequence_mode", False)):
            actions = list(getattr(self, "main_waypoint_actions", []))
            if any(action in {"GRIP", "RELEASE"} for action in actions):
                self._plot_direct_sequence_trajectory(actions)
                return

        segments = getattr(self, "optimized_grip_release_segments", [])
        if len(segments) > 0:
            prefix = np.array(getattr(self, "optimized_prefix_trajectory", []), dtype=float)
            if len(prefix) > 1:
                self.opt_line.set_data(prefix[:, 1], prefix[:, 0])
                self.opt_line.set_3d_properties(prefix[:, 2])
                self.opt_line.set_color("#424242")
                self.opt_line.set_linewidth(2.0)
                self.opt_line.set_alpha(0.85)
            else:
                self.opt_line.set_data([], [])
                self.opt_line.set_3d_properties([])

            show_last_return = getattr(self, "cycle_mode", False)
            for index, segment_entry in enumerate(segments):
                outbound_segment = self._get_segment_path(segment_entry, "outbound")
                return_segment = self._get_segment_path(segment_entry, "return")

                if len(outbound_segment) > 1:
                    outbound_line, = self.ax.plot(
                        outbound_segment[:, 1],
                        outbound_segment[:, 0],
                        outbound_segment[:, 2],
                        color="#d32f2f",
                        linewidth=2.4,
                        alpha=0.95,
                        zorder=5,
                    )
                    self.grip_release_line_artists.append(outbound_line)

                if (show_last_return or index < len(segments) - 1) and len(return_segment) > 1:
                    return_line, = self.ax.plot(
                        return_segment[:, 1],
                        return_segment[:, 0],
                        return_segment[:, 2],
                        color="#1976d2",
                        linewidth=2.4,
                        alpha=0.95,
                        zorder=5,
                    )
                    self.grip_release_line_artists.append(return_line)

            return

        trajectory = np.array(self.optimized_trajectory, dtype=float)
        self.opt_line.set_data(trajectory[:, 1], trajectory[:, 0])
        self.opt_line.set_3d_properties(trajectory[:, 2])
        self.opt_line.set_color("black")
        self.opt_line.set_linewidth(2.5)
        self.opt_line.set_alpha(0.9)

    def _plot_direct_sequence_trajectory(self, actions):
        """Warnai direct trajectory berdasarkan kondisi objek pada gripper."""
        trajectory = np.asarray(self.optimized_trajectory, dtype=float)
        self.opt_line.set_data([], [])
        self.opt_line.set_3d_properties([])

        carrying_object = False
        first_grip_reached = False

        for target_index in range(1, len(trajectory)):
            target_action_index = target_index - 1
            target_action = (
                actions[target_action_index]
                if target_action_index < len(actions)
                else None
            )

            if carrying_object:
                color = "#d32f2f"  # Grip -> waypoint -> Release.
            elif first_grip_reached:
                color = "#1976d2"  # Release -> waypoint -> Grip.
            else:
                color = "#424242"  # Homing/awal -> Grip pertama.

            segment = trajectory[target_index - 1:target_index + 1]
            line, = self.ax.plot(
                segment[:, 1],
                segment[:, 0],
                segment[:, 2],
                color=color,
                linewidth=2.4,
                alpha=0.95,
                zorder=5,
            )
            self.grip_release_line_artists.append(line)

            if target_action == "GRIP":
                carrying_object = True
                first_grip_reached = True
            elif target_action == "RELEASE":
                carrying_object = False

    def _get_segment_path(self, segment_entry, path_name):
        if isinstance(segment_entry, dict):
            path = segment_entry.get(path_name)
            if path is not None and len(path) > 0:
                return np.array(path, dtype=float)

            fallback = segment_entry.get("outbound")
            if fallback is not None and len(fallback) > 0:
                fallback = np.array(fallback, dtype=float)
                return fallback[::-1] if path_name == "return" else fallback

        segment = np.array(segment_entry, dtype=float)
        if path_name == "return":
            return segment[::-1]
        return segment

    def _store_grip_release_segments_from_trajectory(self, source_trajectory):
        self.optimized_prefix_trajectory = []
        self.optimized_grip_release_segments = []

        if self.grip_point is None or len(self.release_points) == 0:
            return False

        source = np.array(source_trajectory, dtype=float)
        if len(source) < 2:
            return False

        grip = np.array(self.grip_point, dtype=float)
        releases = [np.array(point, dtype=float) for point in self.release_points]

        def nearest_index_after(target, start_index):
            start_index = min(max(int(start_index), 0), len(source) - 1)
            distances = np.linalg.norm(source[start_index:] - target, axis=1)
            return start_index + int(np.argmin(distances))

        first_grip_index = nearest_index_after(grip, 0)
        self.optimized_prefix_trajectory = source[:first_grip_index + 1]

        if len(self.optimized_prefix_trajectory) == 0:
            self.optimized_prefix_trajectory = np.array([[0.0, 0.0, 0.0], grip], dtype=float)

        search_start = first_grip_index

        for release in releases:
            grip_index = nearest_index_after(grip, search_start)
            release_index = nearest_index_after(release, grip_index)

            if release_index <= grip_index:
                segment = np.array([grip, release], dtype=float)
            else:
                segment = source[grip_index:release_index + 1].copy()
                segment[0] = grip
                segment[-1] = release

            if len(segment) < 2:
                segment = np.array([grip, release], dtype=float)

            self.optimized_grip_release_segments.append({
                "outbound": segment,
                "return": segment[::-1].copy(),
            })
            search_start = max(release_index, grip_index + 1)

        return len(self.optimized_grip_release_segments) > 0

    def _compose_grip_release_preview_trajectory(self, include_last_return=False):
        if len(self.optimized_grip_release_segments) == 0:
            return np.array([], dtype=float)

        def smooth_segment(segment):
            if len(segment) < 2:
                return np.array(segment, dtype=float)
            
            point_spacing = 0.1
            smooth_trajectory = [segment[0]]
            for i in range(1, len(segment)):
                p1 = smooth_trajectory[-1]
                p2 = segment[i]
                dist = np.linalg.norm(p2 - p1)
                while dist > point_spacing:
                    direction = (p2 - p1) / dist
                    new_point = p1 + direction * point_spacing
                    smooth_trajectory.append(new_point)
                    p1 = new_point
                    dist = np.linalg.norm(p2 - p1)
                smooth_trajectory.append(p2)
                
            arr = np.array(smooth_trajectory)
            window = min(100, len(arr) // 4)
            if window >= 5:
                smoothed = arr.copy()
                for axis in range(3):
                    pad_start = np.repeat(arr[0:1, axis], window)
                    pad_end = np.repeat(arr[-1:, axis], window)
                    padded = np.concatenate((pad_start, arr[:, axis], pad_end))
                    
                    cumsum = np.cumsum(np.insert(padded, 0, 0))
                    ma = (cumsum[window:] - cumsum[:-window]) / float(window)
                    
                    start_idx = (len(ma) - len(arr)) // 2
                    smoothed_axis = ma[start_idx : start_idx + len(arr)]
                    
                    blend = np.linspace(0, 1, window)
                    smoothed_axis[:window] = arr[:window, axis] * (1 - blend) + smoothed_axis[:window] * blend
                    smoothed_axis[-window:] = arr[-window:, axis] * blend + smoothed_axis[-window:] * (1 - blend)
                    
                    smoothed[:, axis] = smoothed_axis
                arr = smoothed
                
            workspace_lower = np.array([0.0, 0.0, 0.0])
            workspace_upper = np.array([
                self.workspace_x_max,
                self.workspace_y_max,
                self.workspace_z_max,
            ])
            arr = np.clip(arr, workspace_lower, workspace_upper)
            return arr

        route = []
        def append_point(point):
            next_point = np.array(point, dtype=float)
            if len(route) == 0 or np.linalg.norm(route[-1] - next_point) > 0.1:
                route.append(next_point)

        for point in smooth_segment(self.optimized_prefix_trajectory):
            append_point(point)

        for index, segment_entry in enumerate(self.optimized_grip_release_segments):
            outbound_segment = self._get_segment_path(segment_entry, "outbound")
            return_segment = self._get_segment_path(segment_entry, "return")

            for point in smooth_segment(outbound_segment)[1:]:
                append_point(point)

            if include_last_return or index < len(self.optimized_grip_release_segments) - 1:
                for point in smooth_segment(return_segment)[1:]:
                    append_point(point)

        return np.array(route, dtype=float)

    def _build_ordered_grip_release_plan(self):
        if len(self.main_waypoints) == 0:
            return None

        entries = [
            {
                "point": np.array(point, dtype=float),
                "action": action,
            }
            for point, action in zip(self.main_waypoints, self.main_waypoint_actions)
        ]

        grip_indices = [
            index for index, entry in enumerate(entries)
            if entry["action"] == "GRIP"
        ]

        if len(grip_indices) == 0:
            return None

        first_grip_index = grip_indices[0]
        grip = entries[first_grip_index]["point"].copy()
        prefix = [np.array([0.0, 0.0, 0.0], dtype=float)]

        for entry in entries[:first_grip_index + 1]:
            prefix.append(entry["point"].copy())

        segments = []
        segment_start = first_grip_index
        release_index = 0

        for index in range(first_grip_index + 1, len(entries)):
            if entries[index]["action"] != "RELEASE":
                continue

            release_index += 1
            outbound_targets = [
                entry["point"].copy()
                for entry in entries[segment_start + 1:index + 1]
            ]

            if len(outbound_targets) == 0:
                outbound_targets = [entries[index]["point"].copy()]

            segments.append({
                "release_index": release_index,
                "release": entries[index]["point"].copy(),
                "outbound_targets": outbound_targets,
            })
            segment_start = index

        if len(segments) == 0:
            return None

        return {
            "grip": grip,
            "prefix": np.array(prefix, dtype=float),
            "segments": segments,
        }

    def _build_direct_grip_release_segments(self):
        self.optimized_prefix_trajectory = []
        self.optimized_grip_release_segments = []

        plan = self._build_ordered_grip_release_plan()
        if plan is None:
            return False

        grip = plan["grip"]
        self.optimized_prefix_trajectory = plan["prefix"]
        self.optimized_grip_release_segments = []

        for segment_plan in plan["segments"]:
            outbound = [grip.copy()]
            outbound.extend([
                np.array(point, dtype=float)
                for point in segment_plan["outbound_targets"]
            ])
            outbound = np.array(outbound, dtype=float)
            release = outbound[-1].copy()
            self.optimized_grip_release_segments.append({
                "outbound": outbound,
                "return": np.array([release, grip], dtype=float),
            })
        return True

    def generate_direct_trajectory(self):
        if len(self.main_waypoints) == 0:
            self.direct_trajectory_segment_distances_mm = np.array([], dtype=float)
            self.direct_trajectory_distance_mm = None
            self.direct_trajectory_axis_distances_mm = np.zeros(3, dtype=float)
            self.log_msg("Tambahkan minimal 1 target terlebih dahulu!")
            return

        self._set_end_effector_visuals_visible(False, redraw=False)
        self._clear_fho_epoch_artists()

        # Mode direct harus mengikuti planning_sequence persis. Tidak ada
        # return otomatis ke Grip lama dan tidak ada pengurutan ulang FHO.
        points = [np.array([0.0, 0.0, 0.0])]
        points.extend([np.array(p, dtype=float) for p in self.main_waypoints])
        self.optimized_trajectory = np.array(points, dtype=float)
        self.optimized_prefix_trajectory = []
        self.optimized_grip_release_segments = []
        self.direct_sequence_mode = True

        # Ukur panjang rute direct dari titik homing sampai seluruh waypoint.
        # Perhitungan dilakukan pada setiap segmen XYZ, sehingga belokan atau
        # gerak maju-mundur tetap tercatat sebagai jarak yang benar-benar
        # ditempuh, bukan hanya jarak lurus titik awal--akhir.
        self.direct_trajectory_segment_distances_mm = self.calculate_segment_distances_mm(
            self.optimized_trajectory
        )
        self.direct_trajectory_distance_mm = self.calculate_trajectory_distance_mm(
            self.optimized_trajectory
        )
        self.direct_trajectory_axis_distances_mm = self.calculate_axis_distances_mm(
            self.optimized_trajectory
        )

        self._plot_optimized_trajectory()
        self.random_plot.set_data([], [])
        self.random_plot.set_3d_properties([])

        self._set_optimization_indicator(False)
        self.canvas.draw_idle()
        axis_distances = self.direct_trajectory_axis_distances_mm
        self.log_msg(
            f"Generate trajectory tanpa optimasi selesai: "
            f"{len(self.main_waypoints)} target, "
            f"{len(self.optimized_trajectory) - 1} segmen, OPTIMASI OFF."
        )
        self.log_msg(
            f"Jarak lintasan tanpa optimasi (jumlah jarak Euclidean XYZ): "
            f"{self.direct_trajectory_distance_mm:.2f} mm | "
            f"per sumbu: X={axis_distances[0]:.2f} mm, "
            f"Y={axis_distances[1]:.2f} mm, Z={axis_distances[2]:.2f} mm."
        )
        for segment_index, segment_distance in enumerate(
            self.direct_trajectory_segment_distances_mm,
            start=1,
        ):
            action = ""
            action_index = segment_index - 1
            actions = getattr(self, "main_waypoint_actions", [])
            if action_index < len(actions) and actions[action_index]:
                action = f" [{actions[action_index]}]"
            self.log_msg(
                f"  Segmen {segment_index} (titik {segment_index - 1} "
                f"-> {segment_index}){action}: "
                f"{float(segment_distance):.2f} mm"
            )

    def _make_base_control_points(self, start_p, num_points, targets=None):
        if targets is None:
            targets = self.main_waypoints

        base_points = []
        current_start = start_p

        for target in targets:
            vec = target - current_start
            for i in range(1, num_points + 1):
                base_points.append(current_start + vec * (i / float(num_points + 1)))
            current_start = target

        return np.array(base_points, dtype=float)

    def _make_flat_line_seed_control_points(self, start_p, num_points, targets=None):
        """Buat seed awal tersebar di seluruh garis dengan control point Z=0."""
        points = self._make_base_control_points(start_p, num_points, targets=targets)
        if len(points) > 0:
            points = np.asarray(points, dtype=float).copy()
            points[:, 2] = 0.0
        return points

    def _is_free_motion_segment(
        self,
        start,
        end,
        include_carried_bottle=True,
        ignored_bottle_bottoms=None,
        sample_mm=None,
    ):
        if sample_mm is None:
            sample_mm = self.FHO_COLLISION_SAMPLE_MM
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        distance = float(np.linalg.norm(end - start))
        sample_count = max(2, int(np.ceil(distance / max(2.0, sample_mm))) + 1)
        for ratio in np.linspace(0.0, 1.0, sample_count):
            point = start + (end - start) * ratio
            if self.is_inside_obstacle(
                point,
                include_bottle=include_carried_bottle,
                ignored_bottle_bottoms=ignored_bottle_bottoms,
            ):
                return False
        return True

    def _build_fho_baseline_polyline(
        self,
        start,
        goal,
        include_carried_bottle=True,
        ignored_bottle_bottoms=None,
        max_control_points=30,
    ):
        """Buat baseline bebas collision dengan detour 3D minimal."""
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        nodes = []
        for point in (start, goal):
            if not nodes or np.linalg.norm(point - nodes[-1]) > 0.1:
                nodes.append(np.asarray(point, dtype=float))
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([self.workspace_x_max, self.workspace_y_max, self.workspace_z_max])

        # Jalur referensi deterministik untuk obstacle yang menghalangi garis
        # langsung. Ia dipakai hanya bila seluruh rute vertikal ini aman; untuk
        # ruang kosong, garis langsung tetap dipertahankan.
        if not self._is_free_motion_segment(
            start,
            goal,
            include_carried_bottle,
            ignored_bottle_bottoms,
            sample_mm=self.FHO_COLLISION_SAMPLE_MM,
        ):
            start_z0 = start.copy()
            goal_z0 = goal.copy()
            start_z0[2] = lower[2]
            goal_z0[2] = lower[2]
            vertical_nodes = [start, start_z0, goal_z0, goal]
            if all(
                self._is_free_motion_segment(
                    vertical_nodes[index],
                    vertical_nodes[index + 1],
                    include_carried_bottle,
                    ignored_bottle_bottoms,
                    sample_mm=self.FHO_COLLISION_SAMPLE_MM,
                )
                for index in range(len(vertical_nodes) - 1)
            ):
                return np.asarray(vertical_nodes, dtype=float)

        for _ in range(max_control_points):
            collision_index = None
            collision_point = None
            for index in range(len(nodes) - 1):
                segment_start = np.asarray(nodes[index], dtype=float)
                segment_end = np.asarray(nodes[index + 1], dtype=float)
                if self._is_free_motion_segment(
                    segment_start,
                    segment_end,
                    include_carried_bottle,
                    ignored_bottle_bottoms,
                    sample_mm=self.FHO_COLLISION_SAMPLE_MM,
                ):
                    continue
                distance = float(np.linalg.norm(segment_end - segment_start))
                count = max(
                    3,
                    int(np.ceil(distance / self.FHO_COLLISION_SAMPLE_MM)) + 1,
                )
                for ratio in np.linspace(0.0, 1.0, count):
                    point = segment_start + (segment_end - segment_start) * ratio
                    if self.is_inside_obstacle(
                        point,
                        include_bottle=include_carried_bottle,
                        ignored_bottle_bottoms=ignored_bottle_bottoms,
                    ):
                        collision_index = index
                        collision_point = point
                        break
                if collision_index is not None:
                    break

            if collision_index is None:
                return np.asarray(nodes, dtype=float)
            if len(nodes) - 2 >= max_control_points:
                break

            segment_start = np.asarray(nodes[collision_index], dtype=float)
            segment_end = np.asarray(nodes[collision_index + 1], dtype=float)
            candidates = []
            direction_vectors = [
                np.array([dx, dy, dz], dtype=float)
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for dz in (-1, 0, 1)
                if (dx, dy, dz) != (0, 0, 0)
            ]
            for radius in np.arange(25.0, 251.0, 25.0):
                for direction in direction_vectors:
                    direction /= np.linalg.norm(direction)
                    candidate = np.clip(collision_point + direction * radius, lower, upper)
                    candidates.append(candidate)

            # Kandidat vertikal deterministik menjaga jalur alternatif di
            # bawah obstacle tetap tersedia tanpa memaksa seluruh trajektori
            # berada pada Z=0.
            for base_point in (segment_start, collision_point, segment_end):
                vertical_candidate = np.asarray(base_point, dtype=float).copy()
                vertical_candidate[2] = lower[2]
                candidates.append(vertical_candidate)

            # Sampel ini hanya memperbaiki satu kandidat dasar FHO sebelum
            # epoch; tidak memiliki populasi, update, atau objective tersendiri.
            for _ in range(500):
                candidate = np.random.uniform(lower, upper)
                candidates.append(candidate)
            bridge_candidates = []
            forward_candidates = []
            for candidate in candidates:
                if self.is_inside_obstacle(
                    candidate,
                    include_bottle=include_carried_bottle,
                    ignored_bottle_bottoms=ignored_bottle_bottoms,
                ):
                    continue
                if not self._is_free_motion_segment(
                    segment_start,
                    candidate,
                    include_carried_bottle,
                    ignored_bottle_bottoms,
                    sample_mm=self.FHO_COLLISION_SAMPLE_MM,
                ):
                    continue
                score = float(
                    np.linalg.norm(candidate - segment_start)
                    + np.linalg.norm(segment_end - candidate)
                )
                if self._is_free_motion_segment(
                    candidate,
                    segment_end,
                    include_carried_bottle,
                    ignored_bottle_bottoms,
                    sample_mm=self.FHO_COLLISION_SAMPLE_MM,
                ):
                    bridge_candidates.append((score, candidate))
                elif np.linalg.norm(candidate - segment_end) < np.linalg.norm(segment_start - segment_end) * 1.25:
                    forward_candidates.append((score, candidate))

            available = bridge_candidates if bridge_candidates else forward_candidates
            if not available:
                break
            _, detour = min(available, key=lambda item: item[0])
            if any(np.linalg.norm(detour - existing) < 1.0 for existing in nodes):
                break
            nodes.insert(collision_index + 1, np.asarray(detour, dtype=float))

        return None

    def _control_points_from_polyline(self, polyline, num_points):
        polyline = np.asarray(polyline, dtype=float)
        lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        if cumulative[-1] <= 1e-9:
            return np.repeat(polyline[:1], num_points, axis=0)

        corner_distances = list(cumulative[1:-1])
        uniform_distances = list(np.linspace(0.0, cumulative[-1], num_points + 2)[1:-1])
        selected = corner_distances[:num_points]
        for distance in uniform_distances:
            if len(selected) >= num_points:
                break
            if all(abs(distance - existing) > 0.5 for existing in selected):
                selected.append(float(distance))
        selected = sorted(selected[:num_points])

        points = []
        for distance in selected:
            segment = min(
                np.searchsorted(cumulative, distance, side="right") - 1,
                len(lengths) - 1,
            )
            ratio = (distance - cumulative[segment]) / max(lengths[segment], 1e-9)
            points.append(polyline[segment] + ratio * (polyline[segment + 1] - polyline[segment]))
        while len(points) < num_points:
            points.append(points[-1] if points else polyline[0])
        return np.asarray(points, dtype=float)

    def _make_feasible_seed_control_points(
        self,
        start_p,
        targets,
        num_points,
        include_carried_bottle=True,
        ignored_bottle_bottoms=None,
        max_baseline_attempts=20,
    ):
        for _ in range(max_baseline_attempts):
            result = []
            segment_start = np.asarray(start_p, dtype=float)
            baseline_valid = True
            for target in targets:
                polyline = self._build_fho_baseline_polyline(
                    segment_start,
                    target,
                    include_carried_bottle=include_carried_bottle,
                    ignored_bottle_bottoms=ignored_bottle_bottoms,
                    max_control_points=num_points,
                )
                if polyline is None:
                    baseline_valid = False
                    break
                segment_control_points = self._control_points_from_polyline(
                    polyline,
                    num_points,
                )
                result.extend(segment_control_points)
                segment_start = np.asarray(target, dtype=float)
            if baseline_valid:
                candidate_points = np.asarray(result, dtype=float)
                candidate_spline, spline_points, _ = self._make_candidate_spline(
                    start_p,
                    candidate_points,
                    num_points,
                    targets=targets,
                )
                if (
                    candidate_spline is not None
                    and spline_points is not None
                    and not self._is_trajectory_outside_workspace(spline_points)
                    and not self.is_spline_colliding(
                        spline_points,
                        include_bottle=include_carried_bottle,
                        ignored_bottle_bottoms=ignored_bottle_bottoms,
                    )
                ):
                    return candidate_points

                # Seed polyline tetap boleh mengawali pencarian ketika kurva
                # quintic memotong obstacle. Kandidat ini akan diberi penalti
                # corner pada objective; ia bukan prioritas di atas B-spline
                # yang valid.
                _, linear_points, _ = self._make_candidate_polyline(
                    start_p,
                    candidate_points,
                    num_points,
                    targets=targets,
                )
                if (
                    linear_points is not None
                    and not self._is_trajectory_outside_workspace(linear_points)
                    and not self.is_spline_colliding(
                        linear_points,
                        include_bottle=include_carried_bottle,
                        ignored_bottle_bottoms=ignored_bottle_bottoms,
                    )
                ):
                    return candidate_points
        return None

    def _target_z_policy(self, start_p, targets):
        # Z tidak dikunci. FHO boleh memilih ketinggian yang lebih pendek dan
        # tetap aman untuk menghindari obstacle.
        return False, float(np.asarray(start_p, dtype=float)[2])

    def _is_trajectory_outside_workspace(self, points, tolerance=1e-6):
        """Periksa apakah ada sampel trajektori di luar batas workspace."""
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            return False
        if points.ndim != 2 or points.shape[1] != 3:
            return True

        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([
            self.workspace_x_max,
            self.workspace_y_max,
            self.workspace_z_max,
        ])
        if not np.all(np.isfinite(points)):
            return True
        return bool(
            np.any(points < (lower - tolerance))
            or np.any(points > (upper + tolerance))
        )

    def _z_motion_penalty(self, start_p, targets, spline_pts, is_flat_z, fixed_z):
        return 0.0

    def _aabb_signed_clearance(self, center_a, size_a, center_b, size_b):
        """Jarak signed dua AABB: positif terpisah, negatif penetrasi."""
        center_a = np.asarray(center_a, dtype=float)
        center_b = np.asarray(center_b, dtype=float)
        size_a = np.asarray(size_a, dtype=float)
        size_b = np.asarray(size_b, dtype=float)
        gaps = np.abs(center_a - center_b) - (size_a + size_b) / 2.0

        # AABB terpisah jika minimal satu sumbu memiliki gap positif. Sumbu
        # lain boleh overlap; jarak antar-box tetap ditentukan oleh gap positif.
        if np.any(gaps > 0.0):
            return float(np.linalg.norm(np.maximum(gaps, 0.0)))

        penetration = -gaps[gaps < 0.0]
        return -float(np.min(penetration))

    def _point_clearance_mm(
        self,
        point,
        include_carried_bottle=True,
        ignored_bottle_bottoms=None,
    ):
        """Clearance geometris gripper/botol terhadap semua obstacle."""
        point = np.asarray(point, dtype=float)
        moving_bodies = [(point, self.gripper_size)]
        if include_carried_bottle:
            moving_bodies.append((self._bottle_center_from_bottom(point), self.bottle_size))

        clearances = []
        for obstacle in getattr(self, "obstacles", []):
            cx, cy, cz, sx, sy, sz = self._obstacle_extents(obstacle)
            obstacle_center = np.array([cx, cy, cz], dtype=float)
            obstacle_size = np.array([sx, sy, sz], dtype=float)
            for body_center, body_size in moving_bodies:
                clearances.append(
                    self._aabb_signed_clearance(
                        body_center,
                        body_size,
                        obstacle_center,
                        obstacle_size,
                    )
                )

        for bottle_bottom in getattr(self, "placed_bottle_bottoms", []):
            if self._is_ignored_bottle_bottom(bottle_bottom, ignored_bottle_bottoms):
                continue
            bottle_center = self._bottle_center_from_bottom(bottle_bottom)
            for body_center, body_size in moving_bodies:
                clearances.append(
                    self._aabb_signed_clearance(
                        body_center,
                        body_size,
                        bottle_center,
                        self.bottle_size,
                    )
                )

        if len(clearances) == 0:
            return np.inf
        return float(np.min(clearances))

    def _trajectory_smoothness_cost(self, spline_pts):
        """Penalti perubahan arah; tetap murni geometris dan tanpa waktu."""
        points = np.asarray(spline_pts, dtype=float)
        if len(points) < 3:
            return 0.0

        vectors = np.diff(points, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        valid = lengths > 1e-9
        if np.count_nonzero(valid) < 2:
            return 0.0

        unit_vectors = vectors[valid] / lengths[valid, None]
        cosines = np.sum(unit_vectors[:-1] * unit_vectors[1:], axis=1)
        angles = np.arccos(np.clip(cosines, -1.0, 1.0))

        # Satu sudut tajam menghasilkan penalti lebih besar daripada
        # perubahan arah yang tersebar secara halus di sepanjang kurva.
        return float(self.FHO_SMOOTHNESS_SCALE_MM * np.sum(angles ** 2))

    def _clearance_penalty(
        self,
        spline_pts,
        include_carried_bottle=True,
        ignored_bottle_bottoms=None,
    ):
        safe_clearance = max(float(self.FHO_SAFE_CLEARANCE_MM), 1e-6)
        violations = []
        for point in np.asarray(spline_pts, dtype=float):
            clearance = self._point_clearance_mm(
                point,
                include_carried_bottle=include_carried_bottle,
                ignored_bottle_bottoms=ignored_bottle_bottoms,
            )
            if np.isfinite(clearance):
                violations.append(max(0.0, safe_clearance - clearance) / safe_clearance)

        if len(violations) == 0:
            return 0.0
        return float(np.mean(np.square(violations)))

    def _geometric_objective(
        self,
        start_p,
        targets,
        spline_pts,
        distance_total,
        include_carried_bottle=True,
        ignored_bottle_bottoms=None,
    ):
        target_values = self.main_waypoints if targets is None else targets
        targets = [np.asarray(target, dtype=float) for target in target_values]
        direct_length = 0.0
        cursor = np.asarray(start_p, dtype=float)
        for target in targets:
            direct_length += float(np.linalg.norm(target - cursor))
            cursor = target

        reference_length = max(direct_length, 1.0)
        smoothness_cost = self._trajectory_smoothness_cost(spline_pts)
        clearance_violation = self._clearance_penalty(
            spline_pts,
            include_carried_bottle=include_carried_bottle,
            ignored_bottle_bottoms=ignored_bottle_bottoms,
        )
        clearance_cost = reference_length * clearance_violation

        return float(
            self.FHO_LENGTH_WEIGHT * distance_total
            + self.FHO_SMOOTHNESS_WEIGHT * smoothness_cost
            + self.FHO_CLEARANCE_WEIGHT * clearance_cost
        )

    def _repair_control_points(self, points, is_flat_z, fixed_z):
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([self.workspace_x_max, self.workspace_y_max, self.workspace_z_max])
        repaired = np.clip(points, lower, upper)

        if is_flat_z:
            repaired[:, 2] = fixed_z

        return repaired

    def _random_control_point(
        self,
        is_flat_z,
        fixed_z,
        ignored_bottle_bottoms=None,
        include_carried_bottle=True,
        max_attempts=250
    ):
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([self.workspace_x_max, self.workspace_y_max, self.workspace_z_max])

        for _ in range(max_attempts):
            point = np.random.uniform(lower, upper)

            if is_flat_z:
                point[2] = fixed_z

            if not self.is_inside_obstacle(
                point,
                include_bottle=include_carried_bottle,
                ignored_bottle_bottoms=ignored_bottle_bottoms,
            ):
                return point

        x_values = np.linspace(lower[0], upper[0], 8)
        y_values = np.linspace(lower[1], upper[1], 8)
        z_values = [fixed_z] if is_flat_z else np.linspace(lower[2], upper[2], 8)

        for x in x_values:
            for y in y_values:
                for z in z_values:
                    point = np.array([x, y, z], dtype=float)

                    if not self.is_inside_obstacle(
                        point,
                        include_bottle=include_carried_bottle,
                        ignored_bottle_bottoms=ignored_bottle_bottoms,
                    ):
                        return point

        return lower.copy()

    def _random_control_points(
        self,
        point_shape,
        is_flat_z,
        fixed_z,
        ignored_bottle_bottoms=None,
        include_carried_bottle=True
    ):
        random_points = np.zeros(point_shape, dtype=float)

        for idx in range(point_shape[0]):
            random_points[idx] = self._random_control_point(
                is_flat_z,
                fixed_z,
                ignored_bottle_bottoms=ignored_bottle_bottoms,
                include_carried_bottle=include_carried_bottle,
            )

        return random_points

    def _random_control_points_near_baseline(
        self,
        baseline_points,
        is_flat_z,
        fixed_z,
        ignored_bottle_bottoms=None,
        include_carried_bottle=True,
        max_attempts=120,
    ):
        baseline_points = np.asarray(baseline_points, dtype=float)
        local_points = baseline_points.copy()
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([self.workspace_x_max, self.workspace_y_max, self.workspace_z_max])
        for index, baseline_point in enumerate(baseline_points):
            for _ in range(max_attempts):
                dimension = 2 if is_flat_z else 3
                direction = np.random.normal(size=dimension)
                direction_norm = np.linalg.norm(direction)
                if direction_norm <= 1e-12:
                    continue
                # Akar berpangkat 1/d menghasilkan sebaran merata di dalam
                # lingkaran/bola, dengan radius total tidak pernah melebihi batas konfigurasi.
                radius = self.FHO_BASELINE_RANDOM_RADIUS_MM * (
                    np.random.rand() ** (1.0 / dimension)
                )
                offset = np.zeros(3, dtype=float)
                offset[:dimension] = direction / direction_norm * radius
                candidate = np.clip(baseline_point + offset, lower, upper)
                if is_flat_z:
                    candidate[2] = fixed_z
                if not self.is_inside_obstacle(
                    candidate,
                    include_bottle=include_carried_bottle,
                    ignored_bottle_bottoms=ignored_bottle_bottoms,
                ) and self._is_free_motion_segment(
                    baseline_point,
                    candidate,
                    include_carried_bottle=include_carried_bottle,
                    ignored_bottle_bottoms=ignored_bottle_bottoms,
                    sample_mm=self.FHO_COLLISION_SAMPLE_MM,
                ):
                    local_points[index] = candidate
                    break
            else:
                # Baseline sudah divalidasi bebas collision, sehingga lebih
                # aman mempertahankan titik elit daripada mengambil titik global.
                local_points[index] = baseline_point

        return local_points

    def _uniform_points_along_polyline(self, trajectory, count):
        trajectory = np.asarray(trajectory, dtype=float)
        if len(trajectory) == 0:
            return np.zeros((count, 3), dtype=float)
        if len(trajectory) == 1:
            return np.repeat(trajectory, count, axis=0)
        lengths = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        if cumulative[-1] <= 1e-9:
            return np.repeat(trajectory[:1], count, axis=0)
        distances = np.linspace(0.0, cumulative[-1], count + 2)[1:-1]
        result = []
        for distance in distances:
            segment = min(
                np.searchsorted(cumulative, distance, side="right") - 1,
                len(lengths) - 1,
            )
            ratio = (distance - cumulative[segment]) / max(lengths[segment], 1e-9)
            result.append(
                trajectory[segment]
                + ratio * (trajectory[segment + 1] - trajectory[segment])
            )
        return np.asarray(result, dtype=float)

    def _trajectory_reference_control_points(self, global_best_spline, targets, num_points):
        trajectory = np.vstack(global_best_spline).T
        references = []
        cursor = 0
        for target in targets:
            if cursor >= len(trajectory) - 1:
                references.extend(np.repeat(trajectory[-1:], num_points, axis=0))
                continue
            remaining = trajectory[cursor:]
            segment_end = cursor + int(
                np.argmin(np.linalg.norm(remaining - np.asarray(target, dtype=float), axis=1))
            )
            segment_end = max(cursor + 1, min(segment_end, len(trajectory) - 1))
            references.extend(
                self._uniform_points_along_polyline(
                    trajectory[cursor:segment_end + 1], num_points
                )
            )
            cursor = segment_end
        return np.asarray(references, dtype=float)

    def _constrain_vector_near_global_trajectory(
        self,
        vector,
        global_best_spline,
        targets,
        num_points,
        point_shape,
        is_flat_z,
        fixed_z,
        ignored_bottle_bottoms=None,
        include_carried_bottle=True,
    ):
        references = self._trajectory_reference_control_points(
            global_best_spline, targets, num_points
        )
        points = self._repair_control_points(
            np.asarray(vector, dtype=float).reshape(point_shape), is_flat_z, fixed_z
        )
        radius_limit = self.FHO_BASELINE_RANDOM_RADIUS_MM
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([self.workspace_x_max, self.workspace_y_max, self.workspace_z_max])

        for index, reference in enumerate(references):
            delta = points[index] - reference
            distance = float(np.linalg.norm(delta))
            if distance > radius_limit:
                points[index] = reference + delta * (radius_limit / distance)
            points[index] = np.clip(points[index], lower, upper)
            if is_flat_z:
                points[index, 2] = fixed_z

            if self.is_inside_obstacle(
                points[index],
                include_bottle=include_carried_bottle,
                ignored_bottle_bottoms=ignored_bottle_bottoms,
            ) or not self._is_free_motion_segment(
                reference,
                points[index],
                include_carried_bottle=include_carried_bottle,
                ignored_bottle_bottoms=ignored_bottle_bottoms,
                sample_mm=self.FHO_COLLISION_SAMPLE_MM,
            ):
                points[index] = reference
                for _ in range(80):
                    direction = np.random.normal(size=3)
                    if is_flat_z:
                        direction[2] = 0.0
                    norm = np.linalg.norm(direction)
                    if norm <= 1e-12:
                        continue
                    radius = radius_limit * (np.random.rand() ** (1.0 / (2 if is_flat_z else 3)))
                    candidate = np.clip(reference + direction / norm * radius, lower, upper)
                    if is_flat_z:
                        candidate[2] = fixed_z
                    if not self.is_inside_obstacle(
                        candidate,
                        include_bottle=include_carried_bottle,
                        ignored_bottle_bottoms=ignored_bottle_bottoms,
                    ) and self._is_free_motion_segment(
                        reference,
                        candidate,
                        include_carried_bottle=include_carried_bottle,
                        ignored_bottle_bottoms=ignored_bottle_bottoms,
                        sample_mm=self.FHO_COLLISION_SAMPLE_MM,
                    ):
                        points[index] = candidate
                        break

        return points.reshape(-1)

    def _repair_control_points_away_from_obstacles(
        self,
        points,
        is_flat_z,
        fixed_z,
        ignored_bottle_bottoms=None,
        include_carried_bottle=True
    ):
        repaired = self._repair_control_points(points, is_flat_z, fixed_z)

        for idx, point in enumerate(repaired):
            if self.is_inside_obstacle(
                point,
                include_bottle=include_carried_bottle,
                ignored_bottle_bottoms=ignored_bottle_bottoms,
            ):
                repaired[idx] = self._random_control_point(
                    is_flat_z,
                    fixed_z,
                    ignored_bottle_bottoms=ignored_bottle_bottoms,
                    include_carried_bottle=include_carried_bottle,
                )

        return repaired

    def _build_nodes(self, start_p, control_points, num_points, targets=None):
        if targets is None:
            targets = self.main_waypoints

        nodes = [start_p]

        for i, target in enumerate(targets):
            nodes.extend(control_points[i * num_points:(i + 1) * num_points])
            nodes.append(target)

        clean_nodes = [nodes[0]]
        for p in nodes[1:]:
            if np.linalg.norm(p - clean_nodes[-1]) > 0.1:
                clean_nodes.append(p)

        return clean_nodes

    def _make_candidate_spline(self, start_p, control_points, num_points, targets=None):
        nodes = self._build_nodes(start_p, control_points, num_points, targets=targets)

        if len(nodes) < 2:
            return None, None, np.inf

        try:
            pts_t = np.array(nodes).T
            spline_degree = min(self.FHO_QUINTIC_SPLINE_DEGREE, len(nodes) - 1)
            # Interpolasi node wajib menjaga grip, waypoint, dan release.
            # Kehalusan dikendalikan oleh objective FHO, bukan smoothing factor
            # tetap yang dapat menggeser target.
            tck, _ = splprep(pts_t, s=0.0, k=spline_degree)
            min_samples = getattr(self, "FHO_MIN_SPLINE_SAMPLES", 240)
            sample_multiplier = getattr(self, "FHO_SPLINE_SAMPLE_MULTIPLIER", 25)
            u_new = np.linspace(0, 1, max(min_samples, len(nodes) * sample_multiplier))
            x_new, y_new, z_new = splev(u_new, tck)
            spline_pts = np.vstack((x_new, y_new, z_new)).T

            segment_vectors = np.diff(spline_pts, axis=0)
            distance_total = np.sum(np.linalg.norm(segment_vectors, axis=1))

            return (x_new, y_new, z_new), spline_pts, distance_total
        except Exception:
            return None, None, np.inf

    def _make_candidate_polyline(self, start_p, control_points, num_points, targets=None):
        nodes = self._build_nodes(start_p, control_points, num_points, targets=targets)
        samples = [np.asarray(nodes[0], dtype=float)]
        for node in nodes[1:]:
            segment_start = samples[-1]
            segment_end = np.asarray(node, dtype=float)
            distance = float(np.linalg.norm(segment_end - segment_start))
            count = max(
                2,
                int(np.ceil(distance / self.FHO_COLLISION_SAMPLE_MM)) + 1,
            )
            samples.extend(
                segment_start + (segment_end - segment_start) * ratio
                for ratio in np.linspace(0.0, 1.0, count)[1:]
            )
        points = np.asarray(samples, dtype=float)
        distance_total = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        return (points[:, 0], points[:, 1], points[:, 2]), points, distance_total

    def _round_polyline_safely(
        self,
        polyline,
        include_carried_bottle=True,
        ignored_bottle_bottoms=None,
    ):
        """Bulatkan corner polyline secara konservatif dan validasi ulang."""
        points = np.asarray(polyline, dtype=float)
        if len(points) < 3:
            return points

        for requested_radius in (30.0, 20.0, 12.0, 6.0, 3.0):
            rounded = [points[0]]
            for index in range(1, len(points) - 1):
                previous = points[index - 1]
                corner = points[index]
                following = points[index + 1]
                incoming = corner - previous
                outgoing = following - corner
                incoming_length = float(np.linalg.norm(incoming))
                outgoing_length = float(np.linalg.norm(outgoing))
                if incoming_length <= 1e-6 or outgoing_length <= 1e-6:
                    rounded.append(corner)
                    continue

                incoming_unit = incoming / incoming_length
                outgoing_unit = outgoing / outgoing_length
                cosine = float(np.dot(incoming_unit, outgoing_unit))
                if cosine >= 0.999:
                    rounded.append(corner)
                    continue

                radius = min(
                    requested_radius,
                    0.35 * incoming_length,
                    0.35 * outgoing_length,
                )
                if radius <= 1.0:
                    rounded.append(corner)
                    continue

                entry = corner - incoming_unit * radius
                exit = corner + outgoing_unit * radius
                rounded.append(entry)
                arc_samples = max(4, int(np.ceil(radius / 4.0)))
                for ratio in np.linspace(0.0, 1.0, arc_samples + 1)[1:]:
                    rounded.append(
                        (1.0 - ratio) ** 2 * entry
                        + 2.0 * (1.0 - ratio) * ratio * corner
                        + ratio ** 2 * exit
                    )

            rounded.append(points[-1])
            rounded = np.asarray(rounded, dtype=float)
            if not self.is_spline_colliding(
                rounded,
                include_bottle=include_carried_bottle,
                ignored_bottle_bottoms=ignored_bottle_bottoms,
            ):
                return rounded

        return points

    def _evaluate_candidate(
        self,
        vector,
        start_p,
        num_points,
        point_shape,
        is_flat_z,
        fixed_z,
        collision_penalty,
        targets=None,
        ignored_bottle_bottoms=None,
        include_carried_bottle=True
    ):
        target_values = self.main_waypoints if targets is None else targets
        target_values = [np.asarray(target, dtype=float) for target in target_values]
        anchor_points = np.vstack([
            np.asarray(start_p, dtype=float),
            *target_values,
        ]) if target_values else np.asarray(start_p, dtype=float).reshape(1, 3)
        if self._is_trajectory_outside_workspace(anchor_points):
            return 0.0, np.inf, None, np.asarray(vector, dtype=float).reshape(point_shape), np.inf

        control_points = self._repair_control_points(
            vector.reshape(point_shape),
            is_flat_z,
            fixed_z
        )

        spline, spline_pts, distance_total = self._make_candidate_spline(
            start_p,
            control_points,
            num_points,
            targets=targets,
        )

        if spline is None:
            return 0.0, np.inf, None, control_points, np.inf

        spline_outside_workspace = self._is_trajectory_outside_workspace(spline_pts)
        spline_collision = self.is_spline_colliding(
            spline_pts,
            include_bottle=include_carried_bottle,
            ignored_bottle_bottoms=ignored_bottle_bottoms,
        )

        if spline_outside_workspace or spline_collision:
            # Jika kurva halus memotong obstacle atau keluar workspace, gunakan
            # polyline hanya sebagai fallback feasible. Penalti perubahan arah
            # membuat FHO tetap memilih B-spline atau polyline dengan corner
            # lebih sedikit ketika keduanya sama-sama aman.
            linear_spline, linear_points, linear_distance = self._make_candidate_polyline(
                start_p,
                control_points,
                num_points,
                targets=targets,
            )
            rounded_points = self._round_polyline_safely(
                linear_points,
                include_carried_bottle=include_carried_bottle,
                ignored_bottle_bottoms=ignored_bottle_bottoms,
            )
            if (
                self._is_trajectory_outside_workspace(rounded_points)
                or self.is_spline_colliding(
                    rounded_points,
                    include_bottle=include_carried_bottle,
                    ignored_bottle_bottoms=ignored_bottle_bottoms,
                )
            ):
                objective = linear_distance + collision_penalty
                fitness = 1.0 / (objective + 1e-9)
                return fitness, objective, None, control_points, linear_distance

            objective = self._geometric_objective(
                start_p,
                targets if targets is not None else self.main_waypoints,
                rounded_points,
                float(np.sum(np.linalg.norm(np.diff(rounded_points, axis=0), axis=1))),
                include_carried_bottle=include_carried_bottle,
                ignored_bottle_bottoms=ignored_bottle_bottoms,
            )
            fitness = 1.0 / (objective + 1e-9)
            rounded_spline = (
                rounded_points[:, 0],
                rounded_points[:, 1],
                rounded_points[:, 2],
            )
            rounded_distance = float(
                np.sum(np.linalg.norm(np.diff(rounded_points, axis=0), axis=1))
            )
            return fitness, objective, rounded_spline, control_points, rounded_distance

        objective = self._geometric_objective(
            start_p,
            targets if targets is not None else self.main_waypoints,
            spline_pts,
            distance_total,
            include_carried_bottle=include_carried_bottle,
            ignored_bottle_bottoms=ignored_bottle_bottoms,
        )

        fitness = 1.0 / (objective + 1e-9)
        return fitness, objective, spline, control_points, distance_total

    def _clip_vector(
        self,
        vector,
        point_shape,
        is_flat_z,
        fixed_z,
        ignored_bottle_bottoms=None,
        include_carried_bottle=True
    ):
        return self._repair_control_points_away_from_obstacles(
            vector.reshape(point_shape),
            is_flat_z,
            fixed_z,
            ignored_bottle_bottoms=ignored_bottle_bottoms,
            include_carried_bottle=include_carried_bottle,
        ).reshape(-1)

    def _clear_fho_epoch_artists(self):
        self._remove_dynamic_artists(getattr(self, "fho_epoch_line_artists", []))
        self._remove_dynamic_artists(getattr(self, "fho_dynamic_obstacle_artists", []))
        self.fho_epoch_line_artists = []
        self.fho_dynamic_obstacle_artists = []

    def _plot_fho_epoch_state(
        self,
        candidate_points,
        candidate_splines,
        valid_flags,
        best_points,
        best_spline,
        epoch,
        best_objective,
        best_distance,
        best_fitness,
        include_carried_bottle=True,
    ):
        self._clear_fho_epoch_artists()

        if getattr(self, "FHO_SHOW_TRAINING_CONTROL_POINTS", False):
            all_points = [
                np.array(points, dtype=float)
                for points in candidate_points
                if points is not None and len(points) > 0
            ]
            if len(all_points) > 0:
                stacked_points = np.vstack(all_points)
                self.random_plot.set_data(stacked_points[:, 1], stacked_points[:, 0])
                self.random_plot.set_3d_properties(stacked_points[:, 2])
            else:
                self.random_plot.set_data([], [])
                self.random_plot.set_3d_properties([])
        else:
            self.random_plot.set_data([], [])
            self.random_plot.set_3d_properties([])

        valid_count = 0
        failed_count = 0

        for spline, is_valid in zip(candidate_splines, valid_flags):
            if spline is None:
                failed_count += 1
                continue

            if is_valid:
                color = "#d32f2f" if include_carried_bottle else "#1976d2"
                alpha = 0.34
                valid_count += 1
            else:
                color = "#ffcdd2" if include_carried_bottle else "#bbdefb"
                alpha = 0.30
                failed_count += 1

            line, = self.ax.plot(
                spline[1],
                spline[0],
                spline[2],
                color=color,
                linewidth=0.8,
                alpha=alpha,
                zorder=2,
            )
            self.fho_epoch_line_artists.append(line)

        if best_spline is not None and best_points is not None:
            self.opt_line.set_data(best_spline[1], best_spline[0])
            self.opt_line.set_3d_properties(best_spline[2])
            self.opt_line.set_color("#b71c1c" if include_carried_bottle else "#0d47a1")
            self.opt_line.set_linewidth(2.8)
            self.opt_line.set_alpha(0.95)
            if getattr(self, "FHO_SHOW_TRAINING_DYNAMIC_OBSTACLES", False):
                self._plot_fho_dynamic_obstacles(best_spline, include_carried_bottle)
            self.log_msg(
                f"FHO Epoch {epoch} | Kandidat OK:{valid_count} Gagal:{failed_count} | "
                f"Jbest:{best_objective:.2f} D:{best_distance:.2f} Fit:{best_fitness:.6f}"
            )
        else:
            self.opt_line.set_data([], [])
            self.opt_line.set_3d_properties([])
            self.log_msg(
                f"FHO Epoch {epoch} | Kandidat OK:{valid_count} Gagal:{failed_count} | "
                "belum ada trajektori valid"
            )

        self.canvas.draw_idle()
        self.root.update()

    def _plot_fho_dynamic_obstacles(self, best_spline, include_carried_bottle):
        self._remove_dynamic_artists(getattr(self, "fho_dynamic_obstacle_artists", []))
        self.fho_dynamic_obstacle_artists = []

        if best_spline is None:
            return

        spline_points = np.vstack(best_spline).T
        if len(spline_points) == 0:
            return

        sample_count = min(6, len(spline_points))
        sample_indexes = np.linspace(0, len(spline_points) - 1, sample_count, dtype=int)

        gx, gy, gz = self.gripper_size
        bx, by, bz = self.bottle_size

        for point in spline_points[sample_indexes]:
            gripper_artist = self.draw_box_obstacle(
                point[0],
                point[1],
                point[2],
                gx,
                gy,
                gz,
                facecolor="#e53935" if include_carried_bottle else "#1e88e5",
                edgecolor="#7f0000" if include_carried_bottle else "#0d47a1",
                alpha=0.13,
                linewidth=0.8,
            )
            self.fho_dynamic_obstacle_artists.append(gripper_artist)

            if include_carried_bottle:
                bottle_center = self._bottle_center_from_bottom(point)
                bottle_artist = self.draw_box_obstacle(
                    bottle_center[0],
                    bottle_center[1],
                    bottle_center[2],
                    bx,
                    by,
                    bz,
                    facecolor="#ff9800",
                    edgecolor="#bf5f00",
                    alpha=0.11,
                    linewidth=0.8,
                )
                self.fho_dynamic_obstacle_artists.append(bottle_artist)

    def _run_fho_for_targets(
        self,
        start_p,
        targets,
        num_points,
        population_size,
        max_epoch,
        collision_penalty,
        log_prefix="",
        ignored_bottle_bottoms=None,
        include_carried_bottle=True
    ):
        targets = [np.array(target, dtype=float) for target in targets]
        segment_endpoints = np.vstack([
            np.asarray(start_p, dtype=float),
            *targets,
        ]) if targets else np.asarray(start_p, dtype=float).reshape(1, 3)
        if self._is_trajectory_outside_workspace(segment_endpoints):
            self.log_msg(
                f"{log_prefix or 'Segmen'}: start/target berada di luar workspace."
            )
            return None
        base_points = self._make_base_control_points(start_p, num_points, targets=targets)

        if len(base_points) == 0:
            return None

        is_flat_z, fixed_z = self._target_z_policy(start_p, targets)
        point_shape = base_points.shape
        dimension = base_points.size

        if self.is_inside_obstacle(
            np.asarray(start_p, dtype=float),
            include_bottle=include_carried_bottle,
            ignored_bottle_bottoms=ignored_bottle_bottoms,
        ):
            self.log_msg(
                f"{log_prefix or 'Segmen'}: titik awal berada di volume collision gripper/botol."
            )
            return None
        for target_index, target in enumerate(targets, start=1):
            if self.is_inside_obstacle(
                target,
                include_bottle=include_carried_bottle,
                ignored_bottle_bottoms=ignored_bottle_bottoms,
            ):
                self.log_msg(
                    f"{log_prefix or 'Segmen'}: target {target_index} {target.tolist()} berada "
                    "di volume collision gripper/botol."
                )
                return None

        feasible_seed = self._make_feasible_seed_control_points(
            start_p,
            targets,
            num_points,
            include_carried_bottle=include_carried_bottle,
            ignored_bottle_bottoms=ignored_bottle_bottoms,
        )
        if feasible_seed is None:
            self.log_msg(
                f"{log_prefix or 'Segmen'}: baseline FHO bebas collision tidak dapat dibentuk; "
                "epoch tidak dijalankan. Tambahkan ruang bebas atau control point."
            )
            return None
        baseline_fit, baseline_objective, baseline_spline, _, baseline_distance = self._evaluate_candidate(
            feasible_seed.reshape(-1),
            start_p,
            num_points,
            point_shape,
            is_flat_z,
            fixed_z,
            collision_penalty,
            targets=targets,
            ignored_bottle_bottoms=ignored_bottle_bottoms,
            include_carried_bottle=include_carried_bottle,
        )
        if baseline_spline is None or baseline_fit <= 0.0:
            self.log_msg(f"{log_prefix or 'Segmen'}: baseline FHO masih collision; epoch dibatalkan.")
            return None
        self.log_msg(
            f"{log_prefix or 'Segmen'}: baseline FHO valid sebelum epoch, "
            f"jarak {baseline_distance:.2f} mm, objective {baseline_objective:.2f}."
        )
        flat_line_seed = self._make_flat_line_seed_control_points(
            start_p,
            num_points,
            targets=targets,
        )
        population = [feasible_seed.reshape(-1)]
        if len(flat_line_seed) > 0 and population_size > 1:
            population.insert(0, flat_line_seed.reshape(-1))

        while len(population) < population_size:
            random_points = self._random_control_points_near_baseline(
                feasible_seed,
                is_flat_z,
                fixed_z,
                ignored_bottle_bottoms=ignored_bottle_bottoms,
                include_carried_bottle=include_carried_bottle,
            )
            population.append(random_points.reshape(-1))
        population = np.array(population)

        global_best_vector = None
        global_best_points = None
        global_best_spline = None
        global_best_fitness = 0.0
        global_best_objective = np.inf
        global_best_distance = np.inf

        for epoch in range(1, max_epoch + 1):
            if not self.is_running:
                break

            fitness = []
            objectives = []
            splines = []
            epoch_splines = []
            valid_flags = []
            repaired_points = []
            distances = []

            for vector in population:
                fit, objective, spline, points, distance_total = self._evaluate_candidate(
                    vector,
                    start_p,
                    num_points,
                    point_shape,
                    is_flat_z,
                    fixed_z,
                    collision_penalty,
                    targets=targets,
                    ignored_bottle_bottoms=ignored_bottle_bottoms,
                    include_carried_bottle=include_carried_bottle,
                )
                fitness.append(fit)
                objectives.append(objective)
                splines.append(spline)
                debug_spline = spline
                if debug_spline is not None and self._is_trajectory_outside_workspace(
                    np.vstack(debug_spline).T
                ):
                    debug_spline = None
                if debug_spline is None and np.isfinite(distance_total):
                    debug_spline, _, _ = self._make_candidate_spline(
                        start_p,
                        points,
                        num_points,
                        targets=targets,
                    )
                    if debug_spline is not None and self._is_trajectory_outside_workspace(
                        np.vstack(debug_spline).T
                    ):
                        debug_spline = None
                epoch_splines.append(debug_spline)
                valid_flags.append(spline is not None)
                repaired_points.append(points)
                distances.append(distance_total)

            fitness = np.array(fitness)
            objectives = np.array(objectives)
            distances = np.array(distances)
            order = np.argsort(fitness)[::-1]
            population = population[order]
            fitness = fitness[order]
            objectives = objectives[order]
            distances = distances[order]
            splines = [splines[i] for i in order]
            epoch_splines = [epoch_splines[i] for i in order]
            valid_flags = [valid_flags[i] for i in order]
            repaired_points = [repaired_points[i] for i in order]

            if splines[0] is not None and fitness[0] > global_best_fitness:
                global_best_fitness = fitness[0]
                global_best_objective = objectives[0]
                global_best_distance = distances[0]
                global_best_vector = population[0].copy()
                global_best_points = repaired_points[0].copy()
                global_best_spline = splines[0]

            main_fire = global_best_vector.copy() if global_best_vector is not None else population[0].copy()
            fire_hawk_count = max(
                1,
                min(self.FHO_FIRE_HAWK_COUNT, population_size - 1),
            )
            fire_hawks = population[:fire_hawk_count]
            prey = population[fire_hawk_count:]
            self._append_training_epoch_csv(
                log_prefix or "segment",
                epoch,
                fitness,
                objectives,
                distances,
                valid_flags,
                repaired_points,
                global_best_points,
                global_best_fitness,
                global_best_objective,
                global_best_distance,
                fire_hawk_count,
            )

            if len(prey) == 0:
                prey = population[1:]

            new_candidates = [main_fire]

            for i, hawk in enumerate(fire_hawks):
                if len(fire_hawks) > 1:
                    other_indexes = [idx for idx in range(len(fire_hawks)) if idx != i]
                    other_hawk = fire_hawks[np.random.choice(other_indexes)]
                else:
                    other_hawk = main_fire

                r1 = np.random.rand(dimension)
                r2 = np.random.rand(dimension)
                # Paper Eq. (6): FH_new = FH + (r1 * GB - r2 * FH_near).
                new_hawk = hawk + (r1 * main_fire - r2 * other_hawk)
                new_candidates.append(
                    self._constrain_vector_near_global_trajectory(
                        new_hawk,
                        global_best_spline,
                        targets,
                        num_points,
                        point_shape,
                        is_flat_z,
                        fixed_z,
                        ignored_bottle_bottoms=ignored_bottle_bottoms,
                        include_carried_bottle=include_carried_bottle,
                    )
                )

            territories = [[] for _ in range(len(fire_hawks))]
            if len(prey) > 0:
                prey_distances = np.linalg.norm(prey[:, None, :] - fire_hawks[None, :, :], axis=2)
                nearest_hawks = np.argmin(prey_distances, axis=1)
            else:
                nearest_hawks = []

            for prey_index, hawk_index in enumerate(nearest_hawks):
                territories[hawk_index].append(prey_index)

            global_safe = np.mean(prey, axis=0) if len(prey) > 0 else main_fire

            for prey_index, prey_vector in enumerate(prey):
                hawk_index = nearest_hawks[prey_index] if len(prey) > 0 else 0
                territory_indexes = territories[hawk_index]
                local_safe = (
                    np.mean(prey[territory_indexes], axis=0)
                    if len(territory_indexes) > 0 else global_safe
                )

                fire_hawk = fire_hawks[hawk_index]
                other_hawk = fire_hawks[np.random.randint(len(fire_hawks))]

                r3 = np.random.rand(dimension)
                r4 = np.random.rand(dimension)
                r5 = np.random.rand(dimension)
                r6 = np.random.rand(dimension)

                # Paper Eq. (7): PR_new = PR + (r3 * FH_l - r4 * SP_l).
                local_flee = prey_vector + (r3 * fire_hawk - r4 * local_safe)
                # Paper Eq. (8): PR_new = PR + (r5 * FH_alter - r6 * SP).
                global_flee = prey_vector + (r5 * other_hawk - r6 * global_safe)

                new_candidates.append(
                    self._constrain_vector_near_global_trajectory(
                        local_flee,
                        global_best_spline,
                        targets,
                        num_points,
                        point_shape,
                        is_flat_z,
                        fixed_z,
                        ignored_bottle_bottoms=ignored_bottle_bottoms,
                        include_carried_bottle=include_carried_bottle,
                    )
                )
                new_candidates.append(
                    self._constrain_vector_near_global_trajectory(
                        global_flee,
                        global_best_spline,
                        targets,
                        num_points,
                        point_shape,
                        is_flat_z,
                        fixed_z,
                        ignored_bottle_bottoms=ignored_bottle_bottoms,
                        include_carried_bottle=include_carried_bottle,
                    )
                )

            # Paper-FHO update pool: GB/main fire + FH_new + PR_new candidates.
            merged_population = np.array(new_candidates)
            merged_fitness = []
            for vector in merged_population:
                fit, _, _, _, _ = self._evaluate_candidate(
                    vector,
                    start_p,
                    num_points,
                    point_shape,
                    is_flat_z,
                    fixed_z,
                    collision_penalty,
                    targets=targets,
                    ignored_bottle_bottoms=ignored_bottle_bottoms,
                    include_carried_bottle=include_carried_bottle,
                )
                merged_fitness.append(fit)

            merged_fitness = np.array(merged_fitness)
            population = merged_population[np.argsort(merged_fitness)[::-1][:population_size]]

            self._plot_fho_epoch_state(
                repaired_points,
                epoch_splines,
                valid_flags,
                global_best_points,
                global_best_spline,
                f"{log_prefix}{epoch}",
                global_best_objective,
                global_best_distance,
                global_best_fitness,
                include_carried_bottle=include_carried_bottle,
            )
            time.sleep(0.01)

        if global_best_spline is None:
            return None

        return {
            "spline": global_best_spline,
            "trajectory": np.vstack(global_best_spline).T,
            "points": global_best_points,
            "fitness": global_best_fitness,
            "objective": global_best_objective,
            "distance": global_best_distance,
        }

    def run_optimization(self):
        if len(self.main_waypoints) == 0:
            self.log_msg("Tambahkan minimal 1 target terlebih dahulu!")
            return

        self.direct_sequence_mode = False
        self._set_end_effector_visuals_visible(False, redraw=False)
        self._clear_fho_epoch_artists()
        self._set_optimization_indicator(False)
        num_points = self.FHO_CONTROL_POINTS_PER_SEGMENT
        population_size = self.FHO_POPULATION_SIZE
        max_epoch = self.FHO_MAX_EPOCH
        collision_penalty = 1_000_000.0

        self.is_running = True
        try:
            training_log_dir = self._create_training_log_dir()
            training_config_path = self._write_training_config(
                training_log_dir,
                collision_penalty,
            )
            self.log_msg(f"Data training disimpan di: {training_log_dir}")
            self.log_msg(f"Parameter training disimpan di: {training_config_path}")
        except Exception as exc:
            self.current_training_log_dir = None
            self.log_msg(f"Gagal membuat folder data training: {exc}")

        self.log_msg(f"\n--- MULAI OPTIMASI FHO ({len(self.main_waypoints)} TARGET) ---")
        self.log_msg(
            f"Fire Hawk Optimizer memakai {num_points} control point per segmen, "
            f"{population_size} kandidat ({self.FHO_FIRE_HAWK_COUNT} Fire Hawk dan "
            f"{population_size - self.FHO_FIRE_HAWK_COUNT} Prey), {max_epoch} epoch."
        )
        self.log_msg(
            f"Jarak sampling pemeriksaan collision: {self.FHO_COLLISION_SAMPLE_MM:g} mm."
        )
        self.log_msg(
            f"Sampling B-spline diperbanyak: min {self.FHO_MIN_SPLINE_SAMPLES} titik, "
            f"atau {self.FHO_SPLINE_SAMPLE_MULTIPLIER}x jumlah node."
        )
        self.log_msg(
            "Sebelum epoch, baseline FHO dibentuk dari detour polyline bebas collision "
            "dengan ketinggian Z detour dipertahankan. Baseline menjadi "
            f"elit awal; kandidat lain diacak lokal dengan radius seragam maksimum "
            f"{self.FHO_BASELINE_RANDOM_RADIUS_MM:g} mm dari baseline."
        )
        self.log_msg(
            "Pada setiap epoch, hasil update Fire Hawk/Prey dibatasi dalam tabung radius "
            f"{self.FHO_BASELINE_RANDOM_RADIUS_MM:g} mm di sekitar trajektori global best terbaru."
        )
        self.log_msg(
            "Populasi awal juga memuat seed garis lurus dengan control point tersebar "
            "di sepanjang garis dan Z control point = 0; baseline bebas collision tetap dipertahankan."
        )
        self.log_msg(
            "Objective geometri: panjang lintasan + perubahan arah kurva + penalti clearance; "
            "collision adalah hard constraint dan polyline bukan solusi akhir."
        )

        start_p = np.array([0.0, 0.0, 0.0])

        ordered_grip_release_plan = self._build_ordered_grip_release_plan()

        if ordered_grip_release_plan is not None:
            grip = ordered_grip_release_plan["grip"]

            self.optimized_prefix_trajectory = ordered_grip_release_plan["prefix"]
            self.optimized_grip_release_segments = []
            self.optimized_trajectory = np.array([], dtype=float)

            release_count = len(ordered_grip_release_plan["segments"])
            self.log_msg(
                f"Mode per-segmen aktif: FHO dibuat satu per satu untuk "
                f"{release_count} pasangan jalur sesuai urutan input."
            )

            last_result = None
            failed_segment = None

            for segment_plan in ordered_grip_release_plan["segments"]:
                if not self.is_running:
                    break

                release_index = int(segment_plan["release_index"])
                outbound_targets = [
                    np.array(point, dtype=float)
                    for point in segment_plan["outbound_targets"]
                ]
                release = np.array(segment_plan["release"], dtype=float)
                outbound_ignored_bottles = []
                return_ignored_bottles = []

                if self.grip_action_point is not None:
                    outbound_ignored_bottles.append(np.array(self.grip_action_point, dtype=float))

                if release_index - 1 < len(getattr(self, "release_action_points", [])):
                    release_action_point = np.array(self.release_action_points[release_index - 1], dtype=float)
                    outbound_ignored_bottles.append(release_action_point)
                    return_ignored_bottles.append(release_action_point)

                self.log_msg(
                    f"\n--- FHO SEGMEN {release_index}/{release_count}: "
                    f"GRIP -> {len(outbound_targets)} target sampai RELEASE {release_index} membawa botol ---"
                )

                outbound_result = self._run_fho_for_targets(
                    grip,
                    outbound_targets,
                    num_points,
                    population_size,
                    max_epoch,
                    collision_penalty,
                    log_prefix=f"S{release_index}A-",
                    ignored_bottle_bottoms=outbound_ignored_bottles,
                    include_carried_bottle=True,
                )

                if outbound_result is None:
                    failed_segment = f"{release_index} pergi"
                    break

                self.log_msg(
                    f"\n--- FHO SEGMEN {release_index}/{release_count}: "
                    f"RELEASE {release_index} -> GRIP tanpa botol ---"
                )

                return_result = self._run_fho_for_targets(
                    release,
                    [grip],
                    num_points,
                    population_size,
                    max_epoch,
                    collision_penalty,
                    log_prefix=f"S{release_index}B-",
                    ignored_bottle_bottoms=return_ignored_bottles,
                    include_carried_bottle=False,
                )

                if return_result is None:
                    failed_segment = f"{release_index} pulang"
                    break

                outbound_segment = outbound_result["trajectory"].copy()
                outbound_segment[0] = grip
                outbound_segment[-1] = release

                return_segment = return_result["trajectory"].copy()
                return_segment[0] = release
                return_segment[-1] = grip

                self.optimized_grip_release_segments.append({
                    "outbound": outbound_segment,
                    "return": return_segment,
                })

                self.optimized_trajectory = self._compose_grip_release_preview_trajectory(
                    include_last_return=self.cycle_mode
                )
                self._plot_optimized_trajectory()
                self.canvas.draw_idle()
                self.root.update()

                self.log_msg(
                    f"Segmen {release_index} selesai | "
                    f"Pergi D:{outbound_result['distance']:.2f} J:{outbound_result['objective']:.2f} "
                    f"Fit:{outbound_result['fitness']:.6f} | "
                    f"Pulang D:{return_result['distance']:.2f} J:{return_result['objective']:.2f} "
                    f"Fit:{return_result['fitness']:.6f}"
                )
                last_result = return_result

            if failed_segment is None and len(self.optimized_grip_release_segments) > 0:
                self._set_optimization_indicator(True)

                self.random_plot.set_data([], [])
                self.random_plot.set_3d_properties([])
                self._clear_fho_epoch_artists()
                self._plot_optimized_trajectory()
                self.canvas.draw_idle()
                self.log_msg(
                    f"Selesai! {len(self.optimized_grip_release_segments)} pasangan segmen "
                    "pergi-pulang terbentuk satu per satu."
                )
            else:
                self._set_optimization_indicator(False)
                if failed_segment is not None:
                    self.log_msg(f"Gagal pada segmen {failed_segment}. Target mungkin tertutup obstacle.")
                else:
                    self.log_msg("Optimasi dihentikan sebelum semua segmen selesai.")

            self.is_running = False
            return

        base_points = self._make_base_control_points(start_p, num_points)

        if len(base_points) == 0:
            self.is_running = False
            self.log_msg("Optimasi gagal: control point kosong.")
            return

        is_flat_z, fixed_z = self._target_z_policy(start_p, self.main_waypoints)
        point_shape = base_points.shape
        dimension = base_points.size

        feasible_seed = self._make_feasible_seed_control_points(
            start_p,
            self.main_waypoints,
            num_points,
        )
        if feasible_seed is None:
            self.is_running = False
            self._set_optimization_indicator(False)
            self.log_msg(
                "Baseline FHO bebas collision tidak dapat dibentuk; epoch tidak dijalankan. "
                "Periksa ruang bebas di sekitar start/target."
            )
            return
        baseline_fit, baseline_objective, baseline_spline, _, baseline_distance = self._evaluate_candidate(
            feasible_seed.reshape(-1),
            start_p,
            num_points,
            point_shape,
            is_flat_z,
            fixed_z,
            collision_penalty,
        )
        if baseline_spline is None or baseline_fit <= 0.0:
            self.is_running = False
            self._set_optimization_indicator(False)
            self.log_msg("Baseline FHO masih collision; epoch tidak dijalankan.")
            return
        self.log_msg(
            f"Baseline FHO valid sebelum epoch: jarak {baseline_distance:.2f} mm, "
            f"objective {baseline_objective:.2f}."
        )
        flat_line_seed = self._make_flat_line_seed_control_points(
            start_p,
            num_points,
            targets=self.main_waypoints,
        )
        population = [feasible_seed.reshape(-1)]
        if len(flat_line_seed) > 0 and population_size > 1:
            population.insert(0, flat_line_seed.reshape(-1))

        while len(population) < population_size:
            random_points = self._random_control_points_near_baseline(
                feasible_seed,
                is_flat_z,
                fixed_z,
            )
            population.append(random_points.reshape(-1))
        population = np.array(population)

        global_best_vector = None
        global_best_points = None
        global_best_spline = None
        global_best_fitness = 0.0
        global_best_objective = np.inf
        global_best_distance = np.inf

        for epoch in range(1, max_epoch + 1):
            if not self.is_running:
                break

            fitness = []
            objectives = []
            splines = []
            epoch_splines = []
            valid_flags = []
            repaired_points = []
            distances = []

            for vector in population:
                fit, objective, spline, points, distance_total = self._evaluate_candidate(
                    vector,
                    start_p,
                    num_points,
                    point_shape,
                    is_flat_z,
                    fixed_z,
                    collision_penalty
                )
                fitness.append(fit)
                objectives.append(objective)
                splines.append(spline)
                debug_spline = spline
                if debug_spline is not None and self._is_trajectory_outside_workspace(
                    np.vstack(debug_spline).T
                ):
                    debug_spline = None
                if debug_spline is None and np.isfinite(distance_total):
                    debug_spline, _, _ = self._make_candidate_spline(
                        start_p,
                        points,
                        num_points,
                    )
                    if debug_spline is not None and self._is_trajectory_outside_workspace(
                        np.vstack(debug_spline).T
                    ):
                        debug_spline = None
                epoch_splines.append(debug_spline)
                valid_flags.append(spline is not None)
                repaired_points.append(points)
                distances.append(distance_total)

            fitness = np.array(fitness)
            objectives = np.array(objectives)
            distances = np.array(distances)
            order = np.argsort(fitness)[::-1]
            population = population[order]
            fitness = fitness[order]
            objectives = objectives[order]
            distances = distances[order]
            splines = [splines[i] for i in order]
            epoch_splines = [epoch_splines[i] for i in order]
            valid_flags = [valid_flags[i] for i in order]
            repaired_points = [repaired_points[i] for i in order]

            if splines[0] is not None and fitness[0] > global_best_fitness:
                global_best_fitness = fitness[0]
                global_best_objective = objectives[0]
                global_best_distance = distances[0]
                global_best_vector = population[0].copy()
                global_best_points = repaired_points[0].copy()
                global_best_spline = splines[0]

            main_fire = global_best_vector.copy() if global_best_vector is not None else population[0].copy()
            fire_hawk_count = max(
                1,
                min(self.FHO_FIRE_HAWK_COUNT, population_size - 1),
            )
            fire_hawks = population[:fire_hawk_count]
            prey = population[fire_hawk_count:]
            self._append_training_epoch_csv(
                "main",
                epoch,
                fitness,
                objectives,
                distances,
                valid_flags,
                repaired_points,
                global_best_points,
                global_best_fitness,
                global_best_objective,
                global_best_distance,
                fire_hawk_count,
            )

            if len(prey) == 0:
                prey = population[1:]

            new_candidates = [main_fire]

            # Paper Eq. (6): Fire Hawk bergerak memakai main fire / GB dan Fire Hawk terdekat lain.
            for i, hawk in enumerate(fire_hawks):
                if len(fire_hawks) > 1:
                    other_indexes = [idx for idx in range(len(fire_hawks)) if idx != i]
                    other_hawk = fire_hawks[np.random.choice(other_indexes)]
                else:
                    other_hawk = main_fire

                r1 = np.random.rand(dimension)
                r2 = np.random.rand(dimension)
                new_hawk = hawk + (r1 * main_fire - r2 * other_hawk)
                new_candidates.append(
                    self._constrain_vector_near_global_trajectory(
                        new_hawk,
                        global_best_spline,
                        self.main_waypoints,
                        num_points,
                        point_shape,
                        is_flat_z,
                        fixed_z,
                    )
                )

            # Prey ditempatkan ke Fire Hawk terdekat untuk membentuk territory.
            territories = [[] for _ in range(len(fire_hawks))]
            if len(prey) > 0:
                distances = np.linalg.norm(prey[:, None, :] - fire_hawks[None, :, :], axis=2)
                nearest_hawks = np.argmin(distances, axis=1)
            else:
                nearest_hawks = []

            for prey_index, hawk_index in enumerate(nearest_hawks):
                territories[hawk_index].append(prey_index)

            global_safe = np.mean(prey, axis=0) if len(prey) > 0 else main_fire

            for prey_index, prey_vector in enumerate(prey):
                hawk_index = nearest_hawks[prey_index] if len(prey) > 0 else 0
                territory_indexes = territories[hawk_index]
                local_safe = (
                    np.mean(prey[territory_indexes], axis=0)
                    if len(territory_indexes) > 0 else global_safe
                )

                fire_hawk = fire_hawks[hawk_index]
                other_hawk = fire_hawks[np.random.randint(len(fire_hawks))]

                r3 = np.random.rand(dimension)
                r4 = np.random.rand(dimension)
                r5 = np.random.rand(dimension)
                r6 = np.random.rand(dimension)

                # Paper Eq. (7): PR_new = PR + (r3 * FH_l - r4 * SP_l).
                local_flee = prey_vector + (r3 * fire_hawk - r4 * local_safe)
                # Paper Eq. (8): PR_new = PR + (r5 * FH_alter - r6 * SP).
                global_flee = prey_vector + (r5 * other_hawk - r6 * global_safe)

                new_candidates.append(
                    self._constrain_vector_near_global_trajectory(
                        local_flee,
                        global_best_spline,
                        self.main_waypoints,
                        num_points,
                        point_shape,
                        is_flat_z,
                        fixed_z,
                    )
                )
                new_candidates.append(
                    self._constrain_vector_near_global_trajectory(
                        global_flee,
                        global_best_spline,
                        self.main_waypoints,
                        num_points,
                        point_shape,
                        is_flat_z,
                        fixed_z,
                    )
                )

            # Paper-FHO update pool: GB/main fire + FH_new + PR_new candidates.
            merged_population = np.array(new_candidates)
            merged_fitness = []
            for vector in merged_population:
                fit, _, _, _, _ = self._evaluate_candidate(
                    vector,
                    start_p,
                    num_points,
                    point_shape,
                    is_flat_z,
                    fixed_z,
                    collision_penalty
                )
                merged_fitness.append(fit)

            merged_fitness = np.array(merged_fitness)
            population = merged_population[np.argsort(merged_fitness)[::-1][:population_size]]

            self._plot_fho_epoch_state(
                repaired_points,
                epoch_splines,
                valid_flags,
                global_best_points,
                global_best_spline,
                epoch,
                global_best_objective,
                global_best_distance,
                global_best_fitness,
                include_carried_bottle=True,
            )
            time.sleep(0.01)

        if global_best_spline is not None:
            source_trajectory = np.vstack(global_best_spline).T
            if self._store_grip_release_segments_from_trajectory(source_trajectory):
                self.optimized_trajectory = self._compose_grip_release_preview_trajectory(
                    include_last_return=self.cycle_mode
                )
                self.log_msg(
                    f"Rute grip-release disimpan sebagai {len(self.optimized_grip_release_segments)} "
                    "pasangan segmen pergi-pulang."
                )
            else:
                self.optimized_trajectory = source_trajectory

            self._set_optimization_indicator(True)
            self.random_plot.set_data([], [])
            self.random_plot.set_3d_properties([])
            self._clear_fho_epoch_artists()
            self._plot_optimized_trajectory()
            self.canvas.draw_idle()
            self.log_msg("Selesai! Jalur multi-target FHO terbentuk.")
        else:
            self._set_optimization_indicator(False)
            self.log_msg("Gagal. Target mungkin tertutup penuh oleh obstacle.")

        self.is_running = False
