import csv
import json
import sys
from pathlib import Path
from tkinter import filedialog

import numpy as np


class CsvPatternMixin:
    TRAJECTORY_BUNDLE_FORMAT = "segment_planner_bundle"
    TRAJECTORY_BUNDLE_VERSION = 2

    @staticmethod
    def _json_value(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        raise TypeError(f"Tipe {type(value).__name__} tidak dapat disimpan ke JSON.")

    @staticmethod
    def _point_array(value):
        if value is None:
            return None
        point = np.asarray(value, dtype=float).reshape(-1)
        if point.size < 3 or not np.all(np.isfinite(point[:3])):
            raise ValueError("Data titik trajektori tidak valid.")
        return point[:3].copy()

    def save_trajectory_bundle(self, file_path):
        """Simpan seluruh keadaan planner dalam CSV bundle yang dapat dipulihkan."""
        trajectory = np.asarray(getattr(self, "optimized_trajectory", []), dtype=float)
        if trajectory.ndim != 2 or trajectory.shape[0] < 2 or trajectory.shape[1] < 3:
            raise ValueError("Buat atau muat trajektori terlebih dahulu.")

        header = [
            "format", "version", "record_type", "group_index", "point_index",
            "action", "x_mm", "y_mm", "z_mm", "payload_json",
        ]

        rows = []

        def add_record(record_type, point=None, group_index="", point_index="", action="", payload=None):
            xyz = self._point_array(point)
            payload_text = "" if payload is None else json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                default=self._json_value,
            )
            rows.append([
                self.TRAJECTORY_BUNDLE_FORMAT,
                self.TRAJECTORY_BUNDLE_VERSION,
                record_type,
                group_index,
                point_index,
                "" if action is None else action,
                "" if xyz is None else f"{xyz[0]:.6f}",
                "" if xyz is None else f"{xyz[1]:.6f}",
                "" if xyz is None else f"{xyz[2]:.6f}",
                payload_text,
            ])

        meta = {
            "current_pos": self._point_array(getattr(self, "current_pos", [0, 0, 0])),
            "cycle_mode": bool(getattr(self, "cycle_mode", False)),
            "is_optimasi_on": bool(getattr(self, "is_optimasi_on", False)),
            "direct_sequence_mode": bool(getattr(self, "direct_sequence_mode", False)),
            "is_gripping": bool(getattr(self, "is_gripping", False)),
            "bottle_bottom": getattr(self, "bottle_bottom", None),
            "grip_point": getattr(self, "grip_point", None),
            "grip_action_point": getattr(self, "grip_action_point", None),
            "end_effector_visuals_visible": bool(
                getattr(self, "end_effector_visuals_visible", True)
            ),
            "servo_grip_angle_deg": 30.0,
            "servo_release_angle_deg": 95.0,
            "speed_xy_mm_s": [
                float(self.speed_x_entry.get()) if hasattr(self, "speed_x_entry") else 80.0,
                float(self.speed_y_entry.get()) if hasattr(self, "speed_y_entry") else 80.0,
                float(self.speed_z_entry.get()) if hasattr(self, "speed_z_entry") else 20.0,
            ],
            "z_deg_per_cm": float(self.z_deg_per_cm_entry.get()) if hasattr(self, "z_deg_per_cm_entry") else 15.0,
            "pulse_per_cm_xy": [
                float(self.pulse_x_entry.get()) if hasattr(self, "pulse_x_entry") else 2800.0,
                float(self.pulse_y_entry.get()) if hasattr(self, "pulse_y_entry") else 1400.0,
            ],
            "grip_release_z_offset": float(getattr(self, "grip_release_z_offset", 1.0)),
            "geometry_profile_version": 3,
            "gripper_size": list(getattr(self, "gripper_size", (220, 60, 34))),
            "bottle_size": list(getattr(self, "bottle_size", (230, 56, 140))),
            "gripper_bottle_overlap_z": float(
                getattr(self, "gripper_bottle_overlap_z", 20.0)
            ),
        }
        add_record("META", payload=meta)

        def add_points(record_type, values, group_index="", actions=None):
            for point_index, point in enumerate(values):
                action = actions[point_index] if actions and point_index < len(actions) else ""
                add_record(record_type, point, group_index, point_index, action)

        add_points("TRAJECTORY", trajectory[:, :3])
        add_points("PREFIX", getattr(self, "optimized_prefix_trajectory", []))
        add_points(
            "MAIN_WAYPOINT",
            getattr(self, "main_waypoints", []),
            actions=list(getattr(self, "main_waypoint_actions", [])),
        )
        add_points("EXTRA_WAYPOINT", getattr(self, "extra_waypoints", []))
        add_points("GRIP_POINT", getattr(self, "grip_points", []))
        add_points("GRIP_ACTION_POINT", getattr(self, "grip_action_points", []))
        add_points("RELEASE_POINT", getattr(self, "release_points", []))
        add_points("RELEASE_ACTION_POINT", getattr(self, "release_action_points", []))
        add_points("PLACED_BOTTLE", getattr(self, "placed_bottle_bottoms", []))

        for segment_index, segment in enumerate(
            getattr(self, "optimized_grip_release_segments", [])
        ):
            outbound = self._get_segment_path(segment, "outbound")
            return_path = self._get_segment_path(segment, "return")
            add_points("SEGMENT_OUTBOUND", outbound, group_index=segment_index)
            add_points("SEGMENT_RETURN", return_path, group_index=segment_index)

        for entry_index, entry in enumerate(getattr(self, "planning_sequence", [])):
            add_record(
                "PLANNING",
                entry.get("point"),
                group_index=entry_index,
                action=entry.get("action"),
                payload={"action_point": entry.get("action_point")},
            )

        user_obstacles = list(getattr(self, "user_fixed_obstacles", []))
        for obstacle_index, obstacle in enumerate(getattr(self, "obstacles", [])):
            cx, cy, cz, sx, sy, sz = self._obstacle_extents(obstacle)
            is_user = any(
                np.allclose(
                    np.asarray(self._obstacle_extents(candidate), dtype=float),
                    np.asarray((cx, cy, cz, sx, sy, sz), dtype=float),
                    atol=1e-6,
                )
                for candidate in user_obstacles
            )
            add_record(
                "OBSTACLE",
                (cx, cy, cz),
                group_index=obstacle_index,
                payload={"size": [sx, sy, sz], "is_user": is_user},
            )

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            writer.writerows(rows)

        return file_path

    def _read_trajectory_bundle(self, file_path):
        csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
        with open(file_path, "r", newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            required = {"format", "version", "record_type", "x_mm", "y_mm", "z_mm"}
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                return None
            rows = list(reader)

        if not rows or rows[0].get("format") != self.TRAJECTORY_BUNDLE_FORMAT:
            return None

        state = {
            "meta": {}, "trajectory": [], "prefix": [], "main_waypoints": [],
            "main_actions": [], "extra_waypoints": [], "grip_points": [],
            "grip_action_points": [], "release_points": [],
            "release_action_points": [], "placed_bottles": [], "planning": [],
            "obstacles": [], "segments": {},
        }

        point_records = {
            "TRAJECTORY": "trajectory", "PREFIX": "prefix",
            "EXTRA_WAYPOINT": "extra_waypoints", "GRIP_POINT": "grip_points",
            "GRIP_ACTION_POINT": "grip_action_points",
            "RELEASE_POINT": "release_points",
            "RELEASE_ACTION_POINT": "release_action_points",
            "PLACED_BOTTLE": "placed_bottles",
        }

        def row_point(row):
            point = np.array(
                [float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])],
                dtype=float,
            )
            if not np.all(np.isfinite(point)):
                raise ValueError("Bundle memuat koordinat non-finite.")
            return point

        for row in rows:
            record_type = row.get("record_type", "").strip().upper()
            payload = json.loads(row.get("payload_json") or "{}")
            if record_type == "META":
                state["meta"] = payload
            elif record_type in point_records:
                state[point_records[record_type]].append(row_point(row))
            elif record_type == "MAIN_WAYPOINT":
                state["main_waypoints"].append(row_point(row))
                state["main_actions"].append(row.get("action") or None)
            elif record_type == "PLANNING":
                point = row_point(row)
                action_point = payload.get("action_point")
                state["planning"].append({
                    "action": row.get("action") or None,
                    "point": point,
                    "action_point": point.copy() if action_point is None else self._point_array(action_point),
                })
            elif record_type == "OBSTACLE":
                center = row_point(row)
                size = np.asarray(payload.get("size", []), dtype=float).reshape(-1)
                if size.size < 3 or np.any(size[:3] <= 0) or not np.all(np.isfinite(size[:3])):
                    raise ValueError("Ukuran obstacle pada bundle tidak valid.")
                state["obstacles"].append({
                    "value": tuple(center.tolist() + size[:3].tolist()),
                    "is_user": bool(payload.get("is_user", True)),
                })
            elif record_type in {"SEGMENT_OUTBOUND", "SEGMENT_RETURN"}:
                segment_index = int(row.get("group_index") or 0)
                segment = state["segments"].setdefault(
                    segment_index, {"outbound": [], "return": []}
                )
                key = "outbound" if record_type.endswith("OUTBOUND") else "return"
                segment[key].append(row_point(row))

        if len(state["trajectory"]) < 2:
            raise ValueError("Bundle tidak memiliki minimal dua titik trajektori.")
        return state

    def _restore_trajectory_bundle(self, state, file_path):
        self._clear_generated_trajectory_state()

        self._remove_dynamic_artists(getattr(self, "fixed_obstacle_artists", []))
        self.fixed_obstacle_artists = []
        self.user_fixed_obstacle_artists = []
        self.obstacles = []
        self.user_fixed_obstacles = []

        for obstacle_entry in state["obstacles"]:
            obstacle = obstacle_entry["value"]
            self.obstacles.append(obstacle)
            artist = self.draw_box_obstacle(*obstacle)
            self.fixed_obstacle_artists.append(artist)
            if obstacle_entry["is_user"]:
                self.user_fixed_obstacles.append(obstacle)
                self.user_fixed_obstacle_artists.append(artist)

        self.main_waypoints = [point.copy() for point in state["main_waypoints"]]
        self.main_waypoint_actions = list(state["main_actions"])
        self.extra_waypoints = [point.copy() for point in state["extra_waypoints"]]
        self.grip_points = [point.copy() for point in state["grip_points"]]
        self.grip_action_points = [point.copy() for point in state["grip_action_points"]]
        self.release_points = [point.copy() for point in state["release_points"]]
        self.release_action_points = [point.copy() for point in state["release_action_points"]]
        self.planning_sequence = state["planning"]
        self.optimized_trajectory = np.asarray(state["trajectory"], dtype=float)
        self.optimized_prefix_trajectory = np.asarray(state["prefix"], dtype=float)
        self.optimized_grip_release_segments = []
        for segment_index in sorted(state["segments"]):
            segment = state["segments"][segment_index]
            self.optimized_grip_release_segments.append({
                "outbound": np.asarray(segment["outbound"], dtype=float),
                "return": np.asarray(segment["return"], dtype=float),
            })

        meta = state["meta"]
        self.grip_point = self._point_array(meta.get("grip_point"))
        if self.grip_point is None and self.grip_points:
            self.grip_point = self.grip_points[-1].copy()
        self.grip_action_point = self._point_array(meta.get("grip_action_point"))
        if self.grip_action_point is None and self.grip_action_points:
            self.grip_action_point = self.grip_action_points[-1].copy()
        self.current_pos = self._point_array(meta.get("current_pos", [0, 0, 0]))
        self.cycle_mode = bool(meta.get("cycle_mode", False))
        self.direct_sequence_mode = bool(meta.get("direct_sequence_mode", False))
        self.is_gripping = bool(meta.get("is_gripping", False))
        self.bottle_bottom = self._point_array(meta.get("bottle_bottom"))
        self.placed_bottle_bottoms = [point.copy() for point in state["placed_bottles"]]
        self.servo_grip_angle_deg = 30.0
        self.servo_release_angle_deg = 95.0
        speed_xy_mm_s = meta.get("speed_xy_mm_s", [80.0, 80.0, 20.0])
        if not isinstance(speed_xy_mm_s, (list, tuple)):
            speed_xy_mm_s = [float(speed_xy_mm_s), float(speed_xy_mm_s), 20.0]
        elif len(speed_xy_mm_s) < 3:
            speed_xy_mm_s = list(speed_xy_mm_s) + [20.0]
        z_deg_per_cm = float(meta.get("z_deg_per_cm", 15.0))
        pulse_per_cm_xy = meta.get("pulse_per_cm_xy", [2800.0, 1400.0])
        self.grip_release_z_offset = float(meta.get("grip_release_z_offset", 1.0))
        self.end_effector_visuals_visible = bool(
            meta.get("end_effector_visuals_visible", True)
        )

        # Bundle lama menyimpan geometri terdahulu dan sebelumnya menimpa
        # konfigurasi aplikasi. Hanya profil geometri baru yang boleh dimuat.
        if int(meta.get("geometry_profile_version", 0)) >= 3:
            if "gripper_size" in meta:
                self.gripper_size = tuple(float(value) for value in meta["gripper_size"][:3])
            if "bottle_size" in meta:
                self.bottle_size = tuple(float(value) for value in meta["bottle_size"][:3])
            self.gripper_bottle_overlap_z = float(
                meta.get("gripper_bottle_overlap_z", 20.0)
            )

        if hasattr(self, "btn_cycle"):
            self.btn_cycle.config(text="CYCLE ON" if self.cycle_mode else "CYCLE OFF")
        if hasattr(self, "grip_release_z_offset_entry"):
            self.grip_release_z_offset_entry.delete(0, "end")
            self.grip_release_z_offset_entry.insert(0, f"{self.grip_release_z_offset:g}")
        for entry_name, value in (
            ("speed_x_entry", float(speed_xy_mm_s[0])),
            ("speed_y_entry", float(speed_xy_mm_s[1])),
            ("speed_z_entry", float(speed_xy_mm_s[2])),
            ("z_deg_per_cm_entry", z_deg_per_cm),
            ("pulse_x_entry", float(pulse_per_cm_xy[0])),
            ("pulse_y_entry", float(pulse_per_cm_xy[1])),
        ):
            if hasattr(self, entry_name):
                entry = getattr(self, entry_name)
                entry.delete(0, "end")
                entry.insert(0, f"{value:g}")

        self.csv_pattern_path = str(Path(file_path).resolve())
        self._set_optimization_indicator(bool(meta.get("is_optimasi_on", False)))
        self._refresh_waypoint_plot()
        self._refresh_pick_release_log()
        self._plot_optimized_trajectory()
        self._set_end_effector_visuals_visible(
            self.end_effector_visuals_visible,
            redraw=False,
        )
        self.update_plot()

    def _read_csv_pattern_points(self, file_path):
        with open(file_path, "r", newline="", encoding="utf-8-sig") as csv_file:
            rows = [
                [cell.strip() for cell in row]
                for row in csv.reader(csv_file)
                if row and any(cell.strip() for cell in row)
            ]

        if not rows:
            raise ValueError("File CSV kosong.")

        header = [cell.lower() for cell in rows[0]]
        known_headers = {"x", "x_mm", "y", "y_mm", "z", "z_mm", "sequence", "index"}
        has_header = any(cell in known_headers for cell in header)
        data_rows = rows[1:] if has_header else rows

        if has_header:
            def find_column(*names):
                for name in names:
                    if name in header:
                        return header.index(name)
                return None

            x_column = find_column("x_mm", "x")
            y_column = find_column("y_mm", "y")
            z_column = find_column("z_mm", "z")
            if x_column is None or y_column is None:
                raise ValueError("Header CSV harus memiliki kolom x_mm/x dan y_mm/y.")
        else:
            x_column = None
            y_column = None
            z_column = None

        points = []
        default_z = float(self.workspace_z_max)

        for row_number, row in enumerate(data_rows, start=2 if has_header else 1):
            try:
                if has_header:
                    x_value = float(row[x_column])
                    y_value = float(row[y_column])
                    z_value = float(row[z_column]) if z_column is not None and row[z_column] else default_z
                elif len(row) >= 4:
                    x_value = float(row[1])
                    y_value = float(row[2])
                    z_value = float(row[3])
                elif len(row) == 3:
                    x_value = float(row[0])
                    y_value = float(row[1])
                    z_value = float(row[2])
                elif len(row) == 2:
                    x_value = float(row[0])
                    y_value = float(row[1])
                    z_value = default_z
                else:
                    raise ValueError
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Data CSV tidak valid pada baris {row_number}.") from exc

            point = np.array([x_value, y_value, z_value], dtype=float)
            if not np.all(np.isfinite(point)):
                raise ValueError(f"Nilai non-finite ditemukan pada baris {row_number}.")
            if not 0.0 <= x_value <= float(self.workspace_x_max):
                raise ValueError(f"X baris {row_number} di luar workspace: {x_value:g} mm.")
            if not 0.0 <= y_value <= float(self.workspace_y_max):
                raise ValueError(f"Y baris {row_number} di luar workspace: {y_value:g} mm.")
            if not 0.0 <= z_value <= float(self.workspace_z_max):
                raise ValueError(f"Z baris {row_number} di luar workspace: {z_value:g} mm.")

            if not points or not np.allclose(points[-1], point, atol=1e-6):
                points.append(point)

        if len(points) < 2:
            raise ValueError("CSV harus berisi minimal dua titik trajektori.")

        return np.asarray(points, dtype=float)

    def load_csv_pattern_file(self, file_path):
        if self.is_running:
            self.log_msg("Tidak bisa memuat CSV saat trajektori sedang berjalan.")
            return False

        try:
            bundle_state = self._read_trajectory_bundle(file_path)
            if bundle_state is not None:
                self._restore_trajectory_bundle(bundle_state, file_path)
                self.log_msg(
                    f"Trajektori lengkap dimuat: {Path(file_path).name} | "
                    f"{len(self.optimized_trajectory)} titik | "
                    f"{len(self.obstacles)} obstacle | "
                    f"{len(self.placed_bottle_bottoms)} objek botol."
                )
                return True
            trajectory = self._read_csv_pattern_points(file_path)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self.log_msg(f"Trajektori gagal dimuat: {exc}")
            return False

        self.clear_data()
        self.optimized_trajectory = trajectory
        self.optimized_prefix_trajectory = []
        self.optimized_grip_release_segments = []
        self.direct_sequence_mode = False
        self.csv_pattern_path = str(Path(file_path).resolve())
        self._set_optimization_indicator(False)
        self._plot_optimized_trajectory()
        self.canvas.draw_idle()

        z_min = float(np.min(trajectory[:, 2]))
        z_max = float(np.max(trajectory[:, 2]))
        self.log_msg(
            f"CSV koordinat lama dimuat: {Path(file_path).name} | "
            f"{len(trajectory)} titik | Z {z_min:.1f}-{z_max:.1f} mm | siap RUN."
        )
        return True

    def load_csv_pattern(self):
        file_path = filedialog.askopenfilename(
            parent=self.root,
            title="Load Trajektori",
            filetypes=[("CSV file", "*.csv"), ("All files", "*.*")],
        )
        if not file_path:
            return False
        return self.load_csv_pattern_file(file_path)
