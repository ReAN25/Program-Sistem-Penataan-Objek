import tkinter as tk

import numpy as np


class ManualControlsMixin:
    def _centered_grip_release_input(self):
        """Ambil posisi input sebagai pusat geometris gripper, bukan salah satu sisinya."""
        return np.array([
            np.clip(self.current_pos[0], 0, self.workspace_x_max),
            np.clip(self.current_pos[1], 0, self.workspace_y_max),
            np.clip(self.current_pos[2], 0, self.workspace_z_max),
        ], dtype=float)

    def _trajectory_point_from_action_point(self, action_point):
        trajectory_point = np.array(action_point, dtype=float).copy()
        trajectory_point[2] = np.clip(
            trajectory_point[2] - self.grip_release_z_offset,
            0,
            self.workspace_z_max,
        )
        return trajectory_point

    def _action_point_from_trajectory_point(self, trajectory_point):
        action_point = np.array(trajectory_point, dtype=float).copy()
        action_point[2] = np.clip(
            action_point[2] + self.grip_release_z_offset,
            0,
            self.workspace_z_max,
        )
        return action_point

    def _manual_collision_options(self):
        trajectory_not_generated = len(getattr(self, "optimized_grip_release_segments", [])) == 0

        if trajectory_not_generated:
            ignored_bottles = list(getattr(self, "placed_bottle_bottoms", []))
        else:
            ignored_bottles = self.get_overlapping_placed_bottle_bottoms(
                self.current_pos,
                self.gripper_size,
            )

        if self.bottle_bottom is not None and not trajectory_not_generated:
            ignored_bottles.extend(
                self.get_overlapping_placed_bottle_bottoms(
                    self._bottle_center_from_bottom(self.bottle_bottom),
                    self.bottle_size,
                )
            )

        include_carried_bottle = self.is_gripping and not trajectory_not_generated

        return include_carried_bottle, ignored_bottles

    def move_pos(self, dx, dy, dz):
        if self.is_running:
            return

        if self.manual_waiting_ack:
            self.log_msg("Tunggu gerakan manual sebelumnya selesai.")
            return

        self._set_end_effector_visuals_visible(True, redraw=False)

        new_pos = np.array([
            np.clip(self.current_pos[0] + dx, 0, self.workspace_x_max),
            np.clip(self.current_pos[1] + dy, 0, self.workspace_y_max),
            np.clip(self.current_pos[2] + dz, 0, self.workspace_z_max),
        ])

        include_carried_bottle, ignored_bottles = self._manual_collision_options()

        if not self.is_inside_obstacle(
            new_pos,
            include_bottle=include_carried_bottle,
            ignored_bottle_bottoms=ignored_bottles,
        ):
            if self.is_arduino_connected():
                self.send_target(new_pos[0], new_pos[1], new_pos[2])
            else:
                self.current_pos = new_pos
                self.update_plot()
                self.log_msg(
                    f"OFFLINE -> posisi grafik X:{new_pos[0]:.1f} "
                    f"Y:{new_pos[1]:.1f} Z:{new_pos[2]:.1f}"
                )
        else:
            self.log_msg("DITOLAK: Koordinat menabrak obstacle.")

    def move_to_typed_pos(self, event=None):
        if self.is_running:
            self.log_msg("Animasi sedang berjalan. Tidak bisa memindahkan manual.")
            return

        try:
            self._set_end_effector_visuals_visible(True, redraw=False)
            new_x = float(self.pos_x_entry.get())
            new_y = float(self.pos_y_entry.get())
            new_z = float(self.pos_z_entry.get())

            new_pos = np.array([
                np.clip(new_x, 0, self.workspace_x_max),
                np.clip(new_y, 0, self.workspace_y_max),
                np.clip(new_z, 0, self.workspace_z_max),
            ])

            include_carried_bottle, ignored_bottles = self._manual_collision_options()

            if not self.is_inside_obstacle(
                new_pos,
                include_bottle=include_carried_bottle,
                ignored_bottle_bottoms=ignored_bottles,
            ):
                self.right_frame.focus_set()
                if self.is_arduino_connected():
                    self.send_target(new_pos[0], new_pos[1], new_pos[2])
                else:
                    self.current_pos = new_pos
                    self.update_plot()
                self.log_msg(
                    f"MDI Absolut -> X:{new_pos[0]:.1f} "
                    f"Y:{new_pos[1]:.1f} Z:{new_pos[2]:.1f}"
                )
            else:
                self.log_msg("DITOLAK: Koordinat menabrak obstacle.")
                self.update_plot()

        except ValueError:
            self.log_msg("Input posisi tidak valid. Gunakan angka.")
            self.update_plot()

    def _refresh_waypoint_plot(self):
        normal_pts = []
        grip_pts = []
        release_pts = []

        for point, action in zip(self.main_waypoints, self.main_waypoint_actions):
            if action == "GRIP":
                grip_pts.append(point)
            elif action == "RELEASE":
                release_pts.append(point)
            else:
                normal_pts.append(point)

        def set_plot(plot, points):
            if len(points) == 0:
                plot.set_data([], [])
                plot.set_3d_properties([])
                return

            pts = np.array(points)
            plot.set_data(pts[:, 1], pts[:, 0])
            plot.set_3d_properties(pts[:, 2])

        set_plot(self.waypoint_plot, normal_pts)
        set_plot(self.grip_waypoint_plot, grip_pts)
        set_plot(self.release_waypoint_plot, release_pts)
        self.canvas.draw_idle()

    def add_waypoint(self, action=None):
        target = self.current_pos.copy()
        if action is None:
            self.extra_waypoints.append(target)
            if not hasattr(self, "planning_sequence"):
                self.planning_sequence = []
            self.planning_sequence.append({
                "action": None,
                "point": target.copy(),
                "action_point": target.copy(),
            })

        self._rebuild_planning_waypoints()
        self._refresh_waypoint_plot()

        action_text = f" | ACTION:{action}" if action else ""
        self.log_msg(f"\n[TARGET {len(self.main_waypoints)} DISET] {target}{action_text}")

    def _format_point(self, point):
        if point is None:
            return "-"
        return f"X:{point[0]:.1f} Y:{point[1]:.1f} Z:{point[2]:.1f}"

    def _refresh_pick_release_log(self):
        if hasattr(self, "grip_pos_var"):
            grip_action_points = getattr(self, "grip_action_points", [])
            grip_points = getattr(self, "grip_points", [])

            if len(grip_action_points) == 0:
                self.grip_pos_var.set("-")
            else:
                self.grip_pos_var.set(
                    f"Grip {len(grip_action_points)} | "
                    f"Aksi {self._format_point(grip_action_points[-1])} | "
                    f"Traj {self._format_point(grip_points[-1])}"
                )

        if hasattr(self, "release_log_text"):
            self.release_log_text.config(state=tk.NORMAL)
            self.release_log_text.delete("1.0", tk.END)

            if len(self.release_points) == 0:
                self.release_log_text.insert(tk.END, "-")
            else:
                for idx, point in enumerate(self.release_points, start=1):
                    action_point = self.release_action_points[idx - 1]
                    self.release_log_text.insert(
                        tk.END,
                        f"{idx}. Aksi {self._format_point(action_point)} | "
                        f"Traj {self._format_point(point)}\n"
                    )

            self.release_log_text.config(state=tk.DISABLED)

    def _set_optimization_indicator(self, is_on):
        self.is_optimasi_on = bool(is_on)

        if not hasattr(self, "optimasi_indicator_label"):
            return

        if self.is_optimasi_on:
            self.optimasi_indicator_label.config(text="OPTIMIZATION ON", bg="#2E7D32", fg="white")
        else:
            self.optimasi_indicator_label.config(text="OPTIMIZATION OFF", bg="#C62828", fg="white")

    def _rebuild_planning_waypoints(self):
        if hasattr(self, "planning_sequence") and len(self.planning_sequence) > 0:
            waypoints = [
                np.array(entry["point"], dtype=float).copy()
                for entry in self.planning_sequence
            ]
            actions = [entry.get("action") for entry in self.planning_sequence]
        else:
            waypoints = [p.copy() for p in self.extra_waypoints]
            actions = [None for _ in self.extra_waypoints]

            if self.grip_point is not None:
                if len(self.release_points) == 0:
                    waypoints.append(self.grip_point.copy())
                    actions.append("GRIP")
                else:
                    for release_point in self.release_points:
                        waypoints.append(self.grip_point.copy())
                        actions.append("GRIP")
                        waypoints.append(release_point.copy())
                        actions.append("RELEASE")

        self.main_waypoints = waypoints
        self.main_waypoint_actions = actions
        self.optimized_trajectory = []
        self.optimized_prefix_trajectory = []
        self.optimized_grip_release_segments = []
        self.direct_sequence_mode = False
        self.direct_trajectory_segment_distances_mm = np.array([], dtype=float)
        self.direct_trajectory_distance_mm = None
        self.direct_trajectory_axis_distances_mm = np.zeros(3, dtype=float)

        self.random_plot.set_data([], [])
        self.random_plot.set_3d_properties([])
        self._clear_fho_epoch_artists()
        self._clear_grip_release_line_artists()
        self.opt_line.set_data([], [])
        self.opt_line.set_3d_properties([])

        self._set_optimization_indicator(False)
        self._refresh_waypoint_plot()
        self._refresh_pick_release_log()

    def _upsert_grip_in_planning_sequence(self):
        if not hasattr(self, "planning_sequence"):
            self.planning_sequence = []

        grip_entry = {
            "action": "GRIP",
            "point": self.grip_point.copy(),
            "action_point": self.grip_action_point.copy(),
        }

        # Grip baru harus mempertahankan urutan input. Jangan mengganti Grip
        # pertama karena satu rute dapat memiliki beberapa pasangan Grip/Release.
        self.planning_sequence.append(grip_entry)

    def _apply_grip_state(self, gripping):
        self.is_gripping = bool(gripping)
        self.bottle_bottom = self.current_pos.copy() if self.is_gripping else None
        self._update_grip_button()

    def _place_bottle_obstacle(self, bottom_point):
        if not hasattr(self, "placed_bottle_bottoms"):
            self.placed_bottle_bottoms = []

        new_bottom = np.array(bottom_point, dtype=float)

        for existing_bottom in self.placed_bottle_bottoms:
            if np.linalg.norm(np.array(existing_bottom) - new_bottom) <= 0.5:
                return False

        self.placed_bottle_bottoms.append(new_bottom)
        return True

    def _update_grip_button(self):
        if not hasattr(self, "btn_grip") or not hasattr(self, "btn_release"):
            return

        self.btn_grip.config(bg="SystemButtonFace", fg="black")
        self.btn_release.config(bg="SystemButtonFace", fg="black")

    def _manual_servo_command(self, command, gripping_state=None):
        if self.is_running:
            self.log_msg("Tidak bisa kontrol servo manual saat RUN aktif.")
            return

        if self.manual_waiting_ack:
            self.log_msg("Tunggu gerakan manual selesai sebelum kontrol servo.")
            return

        if self._send_grip_command(command):
            if gripping_state is not None:
                self._apply_grip_state(gripping_state)
                self.update_plot()

    def manual_servo_grip(self):
        self._manual_servo_command("GRIP", gripping_state=True)

    def manual_servo_release(self):
        self._manual_servo_command("RELEASE", gripping_state=False)

    def set_grip_point(self):
        if self.is_running:
            self.log_msg("Tidak bisa set grip saat trajektori sedang berjalan.")
            return

        if self.manual_waiting_ack:
            self.log_msg("Tunggu gerakan manual selesai sebelum set grip.")
            return

        self._set_end_effector_visuals_visible(True, redraw=False)
        self.grip_action_point = self._centered_grip_release_input()
        self.grip_point = self._trajectory_point_from_action_point(self.grip_action_point)
        if not hasattr(self, "grip_points"):
            self.grip_points = []
        if not hasattr(self, "grip_action_points"):
            self.grip_action_points = []
        self.grip_points.append(self.grip_point.copy())
        self.grip_action_points.append(self.grip_action_point.copy())
        self._upsert_grip_in_planning_sequence()
        added_obstacle = self._place_bottle_obstacle(self.grip_action_point)
        self._apply_grip_state(True)
        self._rebuild_planning_waypoints()
        self.update_plot()
        obstacle_text = " + obstacle botol baru" if added_obstacle else " + obstacle botol sudah ada"
        self.log_msg(
            f"PLAN -> Grip {len(self.grip_points)} input {self._format_point(self.grip_action_point)} "
            f"menjadi trajektori {self._format_point(self.grip_point)}{obstacle_text}"
        )

    def add_release_point(self):
        if self.is_running:
            self.log_msg("Tidak bisa tambah release saat trajektori sedang berjalan.")
            return

        if self.manual_waiting_ack:
            self.log_msg("Tunggu gerakan manual selesai sebelum tambah release.")
            return

        self._set_end_effector_visuals_visible(True, redraw=False)
        release_action_point = self._centered_grip_release_input()
        release_point = self._trajectory_point_from_action_point(release_action_point)
        self.release_action_points.append(release_action_point)
        self.release_points.append(release_point)
        if not hasattr(self, "planning_sequence"):
            self.planning_sequence = []
        self.planning_sequence.append({
            "action": "RELEASE",
            "point": release_point.copy(),
            "action_point": release_action_point.copy(),
        })
        added_obstacle = self._place_bottle_obstacle(release_action_point)
        self._apply_grip_state(False)
        self._rebuild_planning_waypoints()
        self.update_plot()
        obstacle_text = " + obstacle botol baru" if added_obstacle else " + obstacle botol sudah ada"
        self.log_msg(
            f"PLAN -> Release point {len(self.release_points)} ditambah: "
            f"input {self._format_point(release_action_point)} menjadi trajektori "
            f"{self._format_point(release_point)}{obstacle_text}."
        )

    def remove_last_planning_point(self):
        if self.is_running:
            self.log_msg("Tidak bisa menghapus titik saat trajektori sedang berjalan.")
            return

        if self.manual_waiting_ack:
            self.log_msg("Tunggu gerakan manual selesai sebelum menghapus titik.")
            return

        if not getattr(self, "planning_sequence", []):
            self.log_msg("Tidak ada Waypoint/Grip/Release yang dapat dihapus.")
            return

        removed = self.planning_sequence.pop()
        removed_action = removed.get("action")
        removed_point = np.array(removed.get("action_point", removed["point"]), dtype=float)

        if removed_action == "GRIP":
            if self.grip_points:
                self.grip_points.pop()
            if self.grip_action_points:
                self.grip_action_points.pop()
            self.grip_point = self.grip_points[-1].copy() if self.grip_points else None
            self.grip_action_point = (
                self.grip_action_points[-1].copy() if self.grip_action_points else None
            )
            removed_label = "Grip"
        elif removed_action == "RELEASE":
            if self.release_points:
                self.release_points.pop()
            if self.release_action_points:
                self.release_action_points.pop()
            removed_label = "Release"
        else:
            if self.extra_waypoints:
                self.extra_waypoints.pop()
            removed_label = "Waypoint"

        # Susun ulang obstacle botol dari titik aksi yang masih tersimpan.
        self.placed_bottle_bottoms = []
        for bottle_point in list(self.grip_action_points) + list(self.release_action_points):
            self._place_bottle_obstacle(bottle_point)

        # Pulihkan kondisi planning sesuai aksi Grip/Release terakhir yang tersisa.
        gripping = False
        last_grip_action_point = None
        for entry in self.planning_sequence:
            if entry.get("action") == "GRIP":
                gripping = True
                last_grip_action_point = np.array(
                    entry.get("action_point", entry["point"]), dtype=float
                )
            elif entry.get("action") == "RELEASE":
                gripping = False
                last_grip_action_point = None

        self.is_gripping = gripping
        self.bottle_bottom = last_grip_action_point.copy() if last_grip_action_point is not None else None
        self._update_grip_button()

        self._rebuild_planning_waypoints()
        self.refresh_dynamic_boxes()
        self.update_plot()
        self.log_msg(
            f"UNDO -> {removed_label} terakhir dihapus: {self._format_point(removed_point)}"
        )

    def toggle_cycle_mode(self):
        if self.is_running:
            self.log_msg("Tidak bisa mengubah cycle saat trajektori sedang berjalan.")
            return

        self.cycle_mode = not self.cycle_mode

        if hasattr(self, "btn_cycle"):
            if self.cycle_mode:
                self.btn_cycle.config(text="CYCLE ON", bg="SystemButtonFace", fg="black")
            else:
                self.btn_cycle.config(text="CYCLE OFF", bg="SystemButtonFace", fg="black")

        if len(getattr(self, "optimized_grip_release_segments", [])) > 0:
            self.optimized_trajectory = self._compose_grip_release_preview_trajectory(
                include_last_return=self.cycle_mode
            )
            self._plot_optimized_trajectory()
            self.canvas.draw_idle()

        mode_text = "ON" if self.cycle_mode else "OFF"
        self.log_msg(f"CYCLE {mode_text}: mode pengulangan RUN diperbarui tanpa menghapus rute generate.")

    def _clear_generated_trajectory_state(self):
        self.optimized_trajectory = []
        self.optimized_prefix_trajectory = []
        self.optimized_grip_release_segments = []
        self.direct_sequence_mode = False
        self.direct_trajectory_segment_distances_mm = np.array([], dtype=float)
        self.direct_trajectory_distance_mm = None
        self.direct_trajectory_axis_distances_mm = np.zeros(3, dtype=float)
        self._set_optimization_indicator(False)
        self.random_plot.set_data([], [])
        self.random_plot.set_3d_properties([])

        self._clear_fho_epoch_artists()
        self._clear_grip_release_line_artists()
        self.opt_line.set_data([], [])
        self.opt_line.set_3d_properties([])

    def clear_data(self):
        if self.is_running:
            self.log_msg("Cannot clear trajectory while animation is running.")
            return

        self._clear_generated_trajectory_state()
        self.canvas.draw_idle()
        self.log_msg("Generated/FHO trajectory has been cleared. Waypoint, Grip, and Release are kept.")

    def clear_all_waypoints(self):
        if self.is_running:
            self.log_msg("Cannot clear waypoints while animation is running.")
            return

        self._clear_generated_trajectory_state()

        self.main_waypoints = []
        self.main_waypoint_actions = []
        self.planning_sequence = []
        self.extra_waypoints = []
        self.grip_point = None
        self.grip_action_point = None
        self.grip_points = []
        self.grip_action_points = []
        self.release_points = []
        self.release_action_points = []

        self._remove_dynamic_artist(getattr(self, "bottle_box_artist", None))
        self._remove_dynamic_artists(getattr(self, "placed_bottle_artists", []))
        self.bottle_box_artist = None
        self.placed_bottle_bottoms = []
        self.placed_bottle_artists = []
        self.bottle_bottom = None

        self._set_end_effector_visuals_visible(True, redraw=False)
        self._apply_grip_state(False)
        self._refresh_pick_release_log()

        self.waypoint_plot.set_data([], [])
        self.waypoint_plot.set_3d_properties([])
        self.grip_waypoint_plot.set_data([], [])
        self.grip_waypoint_plot.set_3d_properties([])
        self.release_waypoint_plot.set_data([], [])
        self.release_waypoint_plot.set_3d_properties([])

        self.refresh_dynamic_boxes()
        self.canvas.draw_idle()
        self.log_msg("All waypoints, Grip, Release, and trajectory have been cleared.")
