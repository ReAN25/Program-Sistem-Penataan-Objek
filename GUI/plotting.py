import tkinter as tk
from datetime import datetime

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class PlottingMixin:
    def init_3d_plot(self):
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.86)
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.ax.set_xlabel('Y (mm)', fontsize=9, labelpad=4)
        self.ax.set_ylabel('X (mm)', fontsize=9, labelpad=4)
        self.ax.set_zlabel('Z (mm)', fontsize=9, labelpad=4)
        self.ax.tick_params(labelsize=9, pad=1)

        self.point_plot, = self.ax.plot([0], [0], [0], marker='o', markersize=3, color='darkred', zorder=10)
        self.waypoint_plot, = self.ax.plot([], [], [], marker='s', linestyle='None', color='blue', markersize=8, alpha=0.8)
        self.grip_waypoint_plot, = self.ax.plot([], [], [], marker='o', linestyle='None', color='red', markersize=9, alpha=0.95)
        self.release_waypoint_plot, = self.ax.plot([], [], [], marker='s', linestyle='None', color='green', markersize=8, alpha=0.9)
        self.random_plot, = self.ax.plot([], [], [], marker='o', linestyle='None', color='green', markersize=6, alpha=0.9)
        self.opt_line, = self.ax.plot([], [], [], color='black', linewidth=2.5, alpha=0.9)

        self.fixed_obstacle_artists = []
        for obs in self.obstacles:
            self.fixed_obstacle_artists.append(
                self.draw_box_obstacle(*self._obstacle_extents(obs))
            )

        self.gripper_box_artist = None
        self.bottle_box_artist = None
        self.placed_bottle_artists = []
        self.refresh_dynamic_boxes()

        self.ax.set_xlim([self.workspace_y_max, 0])
        self.ax.set_ylim([self.workspace_x_max, 0])
        self.ax.set_zlim([self.workspace_z_max, 0])
        self.reset_plot_perspective(redraw=False)
        self._add_plot_component_description()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _add_plot_component_description(self):
        handles = [
            Line2D([0], [0], color="#424242", linewidth=2.0, label="Homing ke Grip"),
            Line2D([0], [0], color="#d32f2f", linewidth=2.4, label="Grip ke Release"),
            Line2D([0], [0], marker='o', linestyle='None', color='red', markersize=7, label="Titik Grip"),
            Line2D([0], [0], marker='s', linestyle='None', color='blue', markersize=7, label="Waypoint"),
            Line2D([0], [0], marker='s', linestyle='None', color='green', markersize=7, label="Titik Release"),
        ]

        self.plot_component_legend = self.fig.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(0.97, 0.98),
            title="Keterangan Plot",
            fontsize=8,
            title_fontsize=9,
            frameon=True,
            borderpad=0.6,
            labelspacing=0.4,
            handlelength=2.0,
        )
        self.plot_component_legend.get_frame().set_alpha(0.92)

    def _box_vertices(self, x_center, y_center, z_center, x_size, y_size, z_size):
        x0, x1 = x_center - x_size/2, x_center + x_size/2
        y0, y1 = y_center - y_size/2, y_center + y_size/2
        z0, z1 = z_center - z_size/2, z_center + z_size/2

        return np.array([
            [y0, x0, z0], [y0, x1, z0], [y1, x1, z0], [y1, x0, z0],
            [y0, x0, z1], [y0, x1, z1], [y1, x1, z1], [y1, x0, z1]
        ])

    def draw_box_obstacle(
        self,
        x_center,
        y_center,
        z_center,
        x_size,
        y_size,
        z_size,
        facecolor='red',
        edgecolor='darkred',
        alpha=0.3,
        linewidth=1.5,
    ):
        v = self._box_vertices(x_center, y_center, z_center, x_size, y_size, z_size)

        faces = [
            [v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]], 
            [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]], 
            [v[1], v[2], v[6], v[5]], [v[0], v[3], v[7], v[4]]  
        ]

        box = Poly3DCollection(
            faces,
            facecolors=facecolor,
            linewidths=linewidth,
            edgecolors=edgecolor,
            alpha=alpha,
        )
        self.ax.add_collection3d(box)
        return box

    def add_fixed_obstacle(self, lower_corner, dimensions):
        x_min, y_min, z_min = (float(value) for value in lower_corner)
        x_size, y_size, z_size = (float(value) for value in dimensions)

        sizes = ((x_size, "panjang X"), (y_size, "lebar Y"), (z_size, "tinggi Z"))
        for size, name in sizes:
            if size <= 0:
                return False, f"Obstacle ditolak: ukuran {name} harus lebih dari 0."

        limits = (
            (x_min, x_size, self.workspace_x_max, "X"),
            (y_min, y_size, self.workspace_y_max, "Y"),
            (z_min, z_size, self.workspace_z_max, "Z"),
        )
        for start, size, maximum, axis in limits:
            if start < 0 or start + size > maximum:
                return False, (
                    f"Obstacle ditolak: {axis} awal harus >= 0 dan "
                    f"{axis} awal + ukuran harus <= {maximum:g}."
                )

        new_obstacle = (
            x_min + x_size / 2.0,
            y_min + y_size / 2.0,
            z_min + z_size / 2.0,
            x_size,
            y_size,
            z_size,
        )
        for obstacle in self.obstacles:
            existing = self._obstacle_extents(obstacle)
            if np.allclose(existing, new_obstacle, atol=0.5, rtol=0.0):
                return False, "Obstacle ditolak: titik dan ukuran tersebut sudah digunakan."

        self.obstacles.append(new_obstacle)
        self.user_fixed_obstacles.append(new_obstacle)
        artist = self.draw_box_obstacle(*new_obstacle)
        self.fixed_obstacle_artists.append(artist)
        self.user_fixed_obstacle_artists.append(artist)
        self.canvas.draw_idle()
        return True, (
            f"Obstacle fix {len(self.obstacles)} ditambah pada "
            f"sudut bawah X={x_min:g}, Y={y_min:g}, Z={z_min:g}; "
            f"ukuran X+={x_size:g}, Y+={y_size:g}, Z+={z_size:g}."
        )

    def remove_last_user_fixed_obstacle(self):
        if not self.user_fixed_obstacles:
            return False, "Tidak ada obstacle input GUI yang dapat dihapus."

        obstacle = self.user_fixed_obstacles.pop()
        for index in range(len(self.obstacles) - 1, -1, -1):
            if self.obstacles[index] == obstacle:
                del self.obstacles[index]
                break

        artist = self.user_fixed_obstacle_artists.pop()
        self._remove_dynamic_artist(artist)
        if artist in self.fixed_obstacle_artists:
            self.fixed_obstacle_artists.remove(artist)
        self.canvas.draw_idle()

        x_min = obstacle[0] - obstacle[3] / 2.0
        y_min = obstacle[1] - obstacle[4] / 2.0
        z_min = obstacle[2] - obstacle[5] / 2.0
        return True, (
            "Obstacle fix input terakhir dihapus dari sudut bawah "
            f"X={x_min:g}, Y={y_min:g}, Z={z_min:g}."
        )

    def _remove_dynamic_artist(self, artist):
        if artist is None:
            return

        try:
            artist.remove()
        except ValueError:
            pass

    def _remove_dynamic_artists(self, artists):
        for artist in artists:
            self._remove_dynamic_artist(artist)

    def _bottle_center_from_bottom(self, bottom_point):
        bx, by, bz = self.bottle_size
        gx, gy, gz = self.gripper_size
        overlap_z = float(getattr(self, "gripper_bottle_overlap_z", 20.0))
        overlap_z = min(max(overlap_z, 0.0), gz, bz)

        return np.array([
            bottom_point[0],
            bottom_point[1],
            bottom_point[2] + (gz / 2.0) + (bz / 2.0) - overlap_z,
        ])

    def refresh_dynamic_boxes(self):
        self._remove_dynamic_artist(getattr(self, "gripper_box_artist", None))
        self._remove_dynamic_artist(getattr(self, "bottle_box_artist", None))
        self._remove_dynamic_artists(getattr(self, "placed_bottle_artists", []))
        self.placed_bottle_artists = []

        bx, by, bz = self.bottle_size

        for placed_bottom in getattr(self, "placed_bottle_bottoms", []):
            placed_center = self._bottle_center_from_bottom(placed_bottom)
            placed_artist = self.draw_box_obstacle(
                placed_center[0],
                placed_center[1],
                placed_center[2],
                bx,
                by,
                bz,
                facecolor='#4caf50',
                edgecolor='#1b5e20',
                alpha=0.30,
                linewidth=1.2,
            )
            self.placed_bottle_artists.append(placed_artist)

        self.gripper_box_artist = None
        self.bottle_box_artist = None
        if not getattr(self, "end_effector_visuals_visible", True):
            return

        gx, gy, gz = self.gripper_size
        self.gripper_box_artist = self.draw_box_obstacle(
            self.current_pos[0],
            self.current_pos[1],
            self.current_pos[2],
            gx,
            gy,
            gz,
            facecolor='red',
            edgecolor='darkred',
            alpha=0.22,
            linewidth=1.2,
        )

        if self.is_gripping:
            self.bottle_bottom = self.current_pos.copy()

        if self.bottle_bottom is not None:
            bottle_center = self._bottle_center_from_bottom(self.bottle_bottom)
            self.bottle_box_artist = self.draw_box_obstacle(
                bottle_center[0],
                bottle_center[1],
                bottle_center[2],
                bx,
                by,
                bz,
                facecolor='#ff9800',
                edgecolor='#bf5f00',
                alpha=0.28,
                linewidth=1.2,
            )

    def _set_end_effector_visuals_visible(self, visible, redraw=True):
        self.end_effector_visuals_visible = bool(visible)

        if self.end_effector_visuals_visible:
            self.point_plot.set_data([self.current_pos[1]], [self.current_pos[0]])
            self.point_plot.set_3d_properties([self.current_pos[2]])
        else:
            self.point_plot.set_data([], [])
            self.point_plot.set_3d_properties([])

        self.refresh_dynamic_boxes()

        if redraw:
            self.canvas.draw_idle()

    def _obstacle_extents(self, obs):
        if len(obs) == 4:
            cx, cy, cz, size = obs
            return cx, cy, cz, size, size, size

        return obs

    def _boxes_overlap(self, center_a, size_a, center_b, size_b, margin=0.0):
        return (
            abs(center_a[0] - center_b[0]) < (size_a[0] + size_b[0]) / 2.0 + margin and
            abs(center_a[1] - center_b[1]) < (size_a[1] + size_b[1]) / 2.0 + margin and
            abs(center_a[2] - center_b[2]) < (size_a[2] + size_b[2]) / 2.0 + margin
        )

    def _body_collides_with_fixed_obstacles(self, moving_center, moving_size):
        for obs in self.obstacles:
            cx, cy, cz, sx, sy, sz = self._obstacle_extents(obs)

            if self._boxes_overlap(moving_center, moving_size, (cx, cy, cz), (sx, sy, sz), margin=1.0):
                return True

        return False

    def _is_ignored_bottle_bottom(self, bottle_bottom, ignored_bottle_bottoms):
        if ignored_bottle_bottoms is None:
            return False

        for ignored_bottom in ignored_bottle_bottoms:
            if np.linalg.norm(np.array(bottle_bottom) - np.array(ignored_bottom)) <= 0.5:
                return True

        return False

    def _body_collides_with_placed_bottles(self, moving_center, moving_size, ignored_bottle_bottoms=None):
        for bottle_bottom in getattr(self, "placed_bottle_bottoms", []):
            if self._is_ignored_bottle_bottom(bottle_bottom, ignored_bottle_bottoms):
                continue

            bottle_center = self._bottle_center_from_bottom(bottle_bottom)
            if self._boxes_overlap(moving_center, moving_size, bottle_center, self.bottle_size):
                return True

        return False

    def get_overlapping_placed_bottle_bottoms(self, moving_center, moving_size):
        overlapping_bottoms = []

        for bottle_bottom in getattr(self, "placed_bottle_bottoms", []):
            bottle_center = self._bottle_center_from_bottom(bottle_bottom)
            if self._boxes_overlap(moving_center, moving_size, bottle_center, self.bottle_size):
                overlapping_bottoms.append(bottle_bottom)

        return overlapping_bottoms

    def is_inside_obstacle(self, p, include_bottle=None, ignored_bottle_bottoms=None):
        if include_bottle is None:
            include_bottle = self.is_gripping

        if self._body_collides_with_fixed_obstacles(p, self.gripper_size):
            return True

        if self._body_collides_with_placed_bottles(p, self.gripper_size, ignored_bottle_bottoms):
            return True

        if include_bottle:
            bottle_center = self._bottle_center_from_bottom(p)
            if self._body_collides_with_fixed_obstacles(bottle_center, self.bottle_size):
                return True

            if self._body_collides_with_placed_bottles(bottle_center, self.bottle_size, ignored_bottle_bottoms):
                return True

        return False

    def is_spline_colliding(self, spline_points, include_bottle=None, ignored_bottle_bottoms=None):
        for p in spline_points:
            if self.is_inside_obstacle(
                p,
                include_bottle=include_bottle,
                ignored_bottle_bottoms=ignored_bottle_bottoms,
            ):
                return True
        return False

    def update_plot(self):
        if getattr(self, "end_effector_visuals_visible", True):
            self.point_plot.set_data([self.current_pos[1]], [self.current_pos[0]])
            self.point_plot.set_3d_properties([self.current_pos[2]])
        else:
            self.point_plot.set_data([], [])
            self.point_plot.set_3d_properties([])

        self.refresh_dynamic_boxes()
        for entry, val in [(self.pos_x_entry, self.current_pos[0]), 
                           (self.pos_y_entry, self.current_pos[1]), 
                           (self.pos_z_entry, self.current_pos[2])]:
            entry.delete(0, tk.END)
            entry.insert(0, f"{val:.1f}")
        self.canvas.draw_idle()

    def _segments_for_image_export(self):
        """Kumpulkan pasangan segmen untuk ekspor gambar S1A, S1B, dst.

        Pada hasil FHO, setiap item ``optimized_grip_release_segments`` sudah
        memiliki jalur pergi (A) dan pulang (B). Pada mode direct, pasangan
        tersebut dibentuk dari urutan waypoint GRIP/RELEASE. Return terakhir
        tetap dibuat sebagai jalur balik agar pasangan segmen terakhir juga
        dapat didokumentasikan meskipun CYCLE OFF tidak menjalankannya.
        """
        exported = []
        generated_segments = list(
            getattr(self, "optimized_grip_release_segments", [])
        )

        if generated_segments:
            for index, segment_entry in enumerate(generated_segments, start=1):
                for suffix, path_name, description, color in (
                    ("A", "outbound", "Grip ke Release", "#d32f2f"),
                    ("B", "return", "Release ke Grip", "#1976d2"),
                ):
                    path = np.asarray(
                        self._get_segment_path(segment_entry, path_name),
                        dtype=float,
                    )
                    if (
                        path.ndim == 2
                        and path.shape[0] >= 2
                        and path.shape[1] >= 3
                        and np.all(np.isfinite(path[:, :3]))
                    ):
                        exported.append(
                            (f"S{index}{suffix}", path[:, :3], description, color)
                        )
            if exported:
                return exported

        # Mode direct: indeks trajectory = indeks waypoint + 1 karena titik
        # homing (0,0,0) berada di awal array trajectory.
        trajectory = np.asarray(
            getattr(self, "optimized_trajectory", []),
            dtype=float,
        )
        actions = list(getattr(self, "main_waypoint_actions", []))
        if trajectory.ndim != 2 or trajectory.shape[0] < 2 or trajectory.shape[1] < 3:
            return exported

        pairs = []
        active_grip_index = None
        pending_release_index = None
        for waypoint_index, action in enumerate(actions):
            trajectory_index = waypoint_index + 1
            if trajectory_index >= len(trajectory):
                break

            if action == "GRIP":
                if pending_release_index is not None and pairs:
                    pairs[-1]["return"] = trajectory[
                        pending_release_index:trajectory_index + 1, :3
                    ].copy()
                    pending_release_index = None
                active_grip_index = trajectory_index
            elif action == "RELEASE" and active_grip_index is not None:
                outbound = trajectory[active_grip_index:trajectory_index + 1, :3].copy()
                if len(outbound) >= 2:
                    pairs.append({
                        "outbound": outbound,
                        "return": outbound[::-1].copy(),
                    })
                    pending_release_index = trajectory_index
                active_grip_index = None

        for index, pair in enumerate(pairs, start=1):
            for suffix, key, description, color in (
                ("A", "outbound", "Grip ke Release", "#d32f2f"),
                ("B", "return", "Release ke Grip", "#1976d2"),
            ):
                path = np.asarray(pair[key], dtype=float)
                if len(path) >= 2 and np.all(np.isfinite(path[:, :3])):
                    exported.append(
                        (f"S{index}{suffix}", path[:, :3], description, color)
                    )

        return exported

    def _add_segment_box_to_axis(
        self,
        axis,
        obstacle,
        facecolor="red",
        edgecolor="darkred",
        alpha=0.25,
        linewidth=1.0,
    ):
        """Tambahkan box obstacle ke axis gambar segmen, bukan axis utama."""
        vertices = self._box_vertices(*self._obstacle_extents(obstacle))
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
        ]
        axis.add_collection3d(
            Poly3DCollection(
                faces,
                facecolors=facecolor,
                linewidths=linewidth,
                edgecolors=edgecolor,
                alpha=alpha,
            )
        )

    def _draw_segment_export_obstacles(self, axis):
        for obstacle in getattr(self, "obstacles", []):
            self._add_segment_box_to_axis(axis, obstacle)

        for bottom_point in getattr(self, "placed_bottle_bottoms", []):
            center = self._bottle_center_from_bottom(bottom_point)
            self._add_segment_box_to_axis(
                axis,
                (
                    center[0], center[1], center[2],
                    self.bottle_size[0], self.bottle_size[1], self.bottle_size[2],
                ),
                facecolor="#4caf50",
                edgecolor="#1b5e20",
                alpha=0.25,
                linewidth=1.0,
            )

    def _configure_segment_export_axis(
        self,
        axis,
        path,
        color,
        view_title,
        elev,
        azim,
        segment_distance,
        workspace_x,
        workspace_y,
        workspace_z,
    ):
        """Gambar satu jalur pada salah satu sudut pandang ekspor."""
        axis.plot(
            path[:, 1], path[:, 0], path[:, 2],
            color=color,
            linewidth=2.2,
            alpha=0.95,
        )
        axis.scatter(
            [path[0, 1]], [path[0, 0]], [path[0, 2]],
            color="#212121", s=16, marker="o",
        )
        axis.scatter(
            [path[-1, 1]], [path[-1, 0]], [path[-1, 2]],
            color=color, s=20, marker="s",
        )
        self._draw_segment_export_obstacles(axis)

        axis.set_xlabel("Y", fontsize=9, labelpad=3)
        axis.set_ylabel("X", fontsize=9, labelpad=3)
        axis.set_zlabel("Z", fontsize=9, labelpad=3)
        axis.tick_params(labelsize=7, pad=1)
        axis.set_xlim([workspace_y, 0])
        axis.set_ylim([workspace_x, 0])
        axis.set_zlim([workspace_z, 0])
        axis.set_box_aspect((workspace_y, workspace_x, workspace_z))
        axis.view_init(elev=elev, azim=azim)
        axis.set_title(view_title, fontsize=11, pad=6)

    def _save_segment_plot_images(self, timestamp):
        """Simpan satu PNG untuk setiap tampilan setiap jalur A/B.

        Setiap jalur menghasilkan delapan file terpisah: empat sudut
        isometrik dan empat tampak samping dari arah berlawanan. Pemisahan
        file membuat setiap gambar dapat dibuka atau dimasukkan ke laporan
        secara mandiri.
        """
        segments = self._segments_for_image_export()
        if not segments:
            return []

        segment_dir = self.plot_image_dir / f"plot_trajektori_{timestamp}_segments"
        segment_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        workspace_x = float(getattr(self, "workspace_x_max", 1.0))
        workspace_y = float(getattr(self, "workspace_y_max", 1.0))
        workspace_z = float(getattr(self, "workspace_z_max", 1.0))

        views = (
            ("ISO1", "ISO 1", 30, -45),
            ("ISO2", "ISO 2", 30, 45),
            ("ISO3", "ISO 3", 30, 135),
            ("ISO4", "ISO 4", 30, 225),
            ("XZ_1", "Tampak X-Z 1", 0, 0),
            ("XZ_2", "Tampak X-Z 2", 0, 180),
            ("YZ_1", "Tampak Y-Z 1", 0, 90),
            ("YZ_2", "Tampak Y-Z 2", 0, 270),
        )

        for label, path, description, color in segments:
            segment_distance = float(
                np.sum(np.linalg.norm(np.diff(path[:, :3], axis=0), axis=1))
            )
            for view_slug, view_title, elev, azim in views:
                figure = Figure(figsize=(8, 7), dpi=100)
                figure.subplots_adjust(
                    left=0.08,
                    right=0.96,
                    bottom=0.14,
                    top=0.86,
                )
                axis = figure.add_subplot(111, projection="3d")
                self._configure_segment_export_axis(
                    axis,
                    path,
                    color,
                    view_title,
                    elev,
                    azim,
                    segment_distance,
                    workspace_x,
                    workspace_y,
                    workspace_z,
                )
                figure.suptitle(
                    f"{label} - {description} | {view_title} | "
                    f"jarak {segment_distance:.2f} mm",
                    fontsize=13,
                    fontweight="bold",
                )
                figure.legend(
                    handles=[
                        Line2D([0], [0], color=color, linewidth=2.4, label=description),
                        Line2D([0], [0], color="#212121", marker="o", linestyle="None", label="Awal"),
                        Line2D([0], [0], color=color, marker="s", linestyle="None", label="Akhir"),
                    ],
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.015),
                    ncol=3,
                    fontsize=9,
                    frameon=True,
                )

                image_path = segment_dir / f"{label}_{view_slug}.png"
                figure.savefig(image_path, dpi=200, bbox_inches="tight")
                figure.clear()
                saved_paths.append(image_path)

        return saved_paths

    def save_plot_image(self):
        self.plot_image_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = self.plot_image_dir / f"plot_trajektori_{timestamp}.png"
        csv_path = self.plot_image_dir / f"plot_trajektori_{timestamp}.csv"

        self.canvas.draw()
        self.fig.savefig(image_path, dpi=200, bbox_inches="tight")

        segment_paths = []
        try:
            segment_paths = self._save_segment_plot_images(timestamp)
        except (OSError, ValueError, TypeError) as exc:
            self.log_msg(f"Gambar segmen gagal disimpan: {exc}")

        try:
            self.save_trajectory_bundle(csv_path)
        except (OSError, ValueError, TypeError) as exc:
            self.log_msg(f"Gambar plot utama disimpan: {image_path}")
            if segment_paths:
                self.log_msg(
                    f"Gambar {len(segment_paths)} tampilan segmen disimpan di: "
                    f"{segment_paths[0].parent}"
                )
            self.log_msg(f"Data trajektori tidak disimpan: {exc}")
            return

        self.log_msg(f"Gambar plot utama disimpan: {image_path}")
        if segment_paths:
            self.log_msg(
                f"Gambar {len(segment_paths)} tampilan segmen "
                f"({segment_paths[0].stem}-{segment_paths[-1].stem}) "
                f"disimpan di: {segment_paths[0].parent}"
            )
        else:
            self.log_msg("Tidak ada pasangan segmen Grip-Release untuk diekspor.")
        self.log_msg(f"Data lengkap trajektori disimpan: {csv_path}")

    def reset_plot_perspective(self, redraw=True):
        self.ax.view_init(elev=self.default_plot_elev, azim=self.default_plot_azim)
        self.ax.set_box_aspect((self.workspace_y_max, self.workspace_x_max, self.workspace_z_max))

        if redraw:
            self.canvas.draw_idle()
            self.log_msg("Perspektif plot dikembalikan ke tampilan default.")

if __name__ == "__main__":
    root = tk.Tk()
    app = Cartesian3DApp(root)
    root.mainloop()
