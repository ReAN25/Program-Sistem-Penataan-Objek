from pathlib import Path
import threading

import numpy as np

from csv_patterns import CsvPatternMixin
from pulse_execution import PulseBlockExecutionMixin
from layout_controls import LayoutControlsMixin
from manual_controls import ManualControlsMixin
from optimization import OptimizationMixin
from plotting import PlottingMixin
from pulse_serial import PulseSerialMixin


class Cartesian3DApp(
    PulseSerialMixin,
    CsvPatternMixin,
    LayoutControlsMixin,
    PlottingMixin,
    ManualControlsMixin,
    OptimizationMixin,
    PulseBlockExecutionMixin,
):
    def __init__(self, root):
        self.root = root
        self.root.title("Trajektori Multi-Target - Pulse Block Streaming XY + Servo")
        self.root.state('zoomed')
        self.root.configure(bg="white")

        self.current_pos = np.array([0.0, 0.0, 0.0])
        self.workspace_x_max = 580.0
        self.workspace_y_max = 500.0
        self.workspace_z_max = 300.0
        # Parameter geometri RUN disamakan dengan Segment Planner. Pulse
        # Block hanya mengganti cara eksekusi waktunya menjadi timeline blok.
        self.run_trajectory_spacing_mm = 0.3
        self.run_points_per_cm = 10.0 / self.run_trajectory_spacing_mm
        self.circle_smoothing_enabled = True
        # Timeline Pulse Block: 10 ms per micro-slice, 2 slice/20 ms per blok,
        # dengan 32 slot resident pada firmware (cadangan sekitar 640 ms).
        self.pulse_slice_ms = 10
        self.pulse_block_ms = 20
        # X/Y memakai DDA pulsa. Z memakai target sudut AS5600 multi-turn
        # pada poros output; rasio 20:1 hanya dipakai pada pulsa motor.
        self.servo_grip_angle_deg = 30
        self.servo_release_angle_deg = 95
        self.is_running = False
        self.stop_requested = False
        
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
        self.cycle_mode = False
        self.is_optimasi_on = False
        self.optimized_trajectory = [] 
        self.optimized_prefix_trajectory = []
        self.optimized_grip_release_segments = []
        self.direct_sequence_mode = False
        # Metrik lintasan hasil Generate Trajectory (tanpa FHO). Nilai ini
        # diisi sebagai jumlah jarak tiap segmen XYZ saat tombol generate
        # ditekan dan dikosongkan kembali ketika rute diubah/dihapus.
        self.direct_trajectory_segment_distances_mm = np.array([], dtype=float)
        self.direct_trajectory_distance_mm = None
        self.direct_trajectory_axis_distances_mm = np.zeros(3, dtype=float)

        # ===== OBSTACLE TETAP (FIXED) =====
        self.obstacles = []
        self.user_fixed_obstacles = []
        self.user_fixed_obstacle_artists = []
        # Koordinat Grip/Release adalah pusat geometris objek pada ketiga sumbu.
        # Lebar X 240 mm berarti bentang objek adalah titik X +/- 120 mm.
        self.gripper_size = (220.0, 60.0, 34.0)
        # Ukuran total obstacle botol pada sumbu X, Y, dan Z.
        self.bottle_size = (230.0, 56.0, 140.0)
        # Kedalaman bagian gripper yang masuk ke volume botol pada sumbu Z.
        self.gripper_bottle_overlap_z = 20.0
        self.grip_release_z_offset = 1.0
        self.is_gripping = False
        self.bottle_bottom = None
        self.placed_bottle_bottoms = []
        self.placed_bottle_artists = []
        self.end_effector_visuals_visible = True
        self.ser = None
        # Satu transaksi perintah/jawaban serial pada satu waktu. Ini mencegah
        # ACK konfigurasi terbaca oleh thread gerak atau pemulihan, dan sebaliknya.
        self.serial_command_lock = threading.RLock()
        self._hardware_pos = np.array([0.0, 0.0, 0.0], dtype=float)
        self.manual_waiting_ack = False
        self.servo_action_sequence = 0
        self.homing_waiting_ack = False
        self.homing_last_tx = 0.0
        self.homing_active_seen = False
        self.homing_retry_count = 0
        self._active_streamer = None
        self.connection_enabled = False
        self.default_plot_elev = 30
        self.default_plot_azim = -60
        self.plot_image_dir = Path(__file__).resolve().parents[1] / "saved_plot_images"
        self.training_log_root = Path(__file__).resolve().parents[1] / "data_training"
        self.run_log_dir = Path(__file__).resolve().parents[1] / "run_logs"
        self.position_state_path = self.run_log_dir / "last_robot_position.json"

        self.create_layout()
        self._load_position_memory()
        self.init_3d_plot()
        self.update_status(False, "Mode Offline - Arduino belum terhubung")
        self.connection_monitor()
