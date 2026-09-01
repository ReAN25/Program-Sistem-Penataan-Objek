import sys
import tempfile
import unittest
import binascii
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "full_gui"))
from pulse_execution import PulseBlockExecutionMixin
from pulse_planner import plan_polyline
from pulse_serial import PulseSerialMixin
from manual_controls import ManualControlsMixin


class DummyRoute(PulseBlockExecutionMixin):
    def __init__(self):
        grip = np.array([10.0, 10.0, 20.0])
        release1 = np.array([50.0, 20.0, 20.0])
        release2 = np.array([70.0, 40.0, 20.0])
        self.grip_point = grip
        self.optimized_prefix_trajectory = np.array([[0, 0, 0], grip])
        self.optimized_grip_release_segments = [
            {"outbound": np.array([grip, release1]), "return": np.array([release1, grip])},
            {"outbound": np.array([grip, release2]), "return": np.array([release2, grip])},
        ]

    @staticmethod
    def _get_segment_path(entry, name):
        return np.asarray(entry[name], dtype=float)


class FullGuiRouteTest(unittest.TestCase):
    def test_direct_route_preserves_internal_grip_release_checkpoints(self):
        class DummyDirect(PulseBlockExecutionMixin):
            def __init__(self):
                self.optimized_trajectory = np.array([
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 10.0],
                    [40.0, 0.0, 10.0],
                    [40.0, 0.0, 20.0],
                ])
                self.main_waypoints = [
                    np.array([0.0, 0.0, 10.0]),
                    np.array([40.0, 0.0, 20.0]),
                ]
                self.main_waypoint_actions = ["GRIP", "RELEASE"]
                self.direct_sequence_mode = True
                self.run_trajectory_spacing_mm = 0.3

        route, action_map = DummyDirect()._route_from_trajectory(False)
        events = [
            (index, action)
            for index in sorted(action_map)
            for action in action_map[index]
        ]
        self.assertEqual([action for _, action in events], ["GRIP", "RELEASE"])
        self.assertEqual(route[events[0][0]].tolist(), [0.0, 0.0, 10.0])
        self.assertEqual(route[events[1][0]].tolist(), [40.0, 0.0, 20.0])
        self.assertLess(events[0][0], events[1][0])

    def test_run_action_moves_to_z_offset_and_returns_to_route(self):
        class Entry:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        class DummyAction(ManualControlsMixin, PulseBlockExecutionMixin, PulseSerialMixin):
            def __init__(self):
                self.workspace_x_max = 580.0
                self.workspace_y_max = 500.0
                self.workspace_z_max = 300.0
                self.grip_release_z_offset = 1.0
                self.grip_release_z_offset_entry = Entry("20")
                self._hardware_pos = np.array([10.0, 20.0, 40.0])
                self.is_running = True
                self.streams = []
                self.root = type("Root", (), {"after": lambda *_args: None})()
                self.log_messages = []

            def log_msg(self, message):
                self.log_messages.append(message)

            def _stream_points_blocking(self, points, progress=None):
                self.streams.append(np.asarray(points, dtype=float).copy())
                self._hardware_pos = np.asarray(points[-1], dtype=float).copy()
                return object()

            def _wait_servo_ack(self, _action):
                return True

            def _place_bottle_obstacle(self, _point):
                return True

            def _apply_grip_state(self, _gripping):
                return None

            def update_plot(self):
                return None

        dummy = DummyAction()
        route_point = np.array([10.0, 20.0, 40.0])
        self.assertTrue(dummy._execute_action("GRIP", route_point))
        self.assertEqual(len(dummy.streams), 2)
        np.testing.assert_allclose(dummy.streams[0][-1], [10.0, 20.0, 60.0])
        np.testing.assert_allclose(dummy.streams[1][-1], route_point)
        self.assertAlmostEqual(dummy.grip_release_z_offset, 20.0)
        self.assertIn("offset +20.0 mm", dummy.log_messages[0])

    def test_manual_recovery_stops_stale_buffer_wait(self):
        class FakeStreamer:
            def __init__(self):
                self.writes = []
                self.lines = ["STREAM_ABORTED"]

            def _write(self, command):
                self.writes.append(command)

            def _wait_for(self, predicate, _timeout):
                while self.lines:
                    line = self.lines.pop(0)
                    if predicate(line):
                        return line
                raise RuntimeError("respons pengujian habis")

        fake = FakeStreamer()
        dummy = object.__new__(PulseSerialMixin)
        dummy._make_streamer = lambda: fake
        positions = iter([
            (np.array([100, 0]), np.array([1.0, 0.0, 20.0]), {"RUN": "1", "XYWAIT": "1", "HOME": "1", "ZENC": "1"}),
            (np.array([100, 0]), np.array([1.0, 0.0, 20.0]), {"RUN": "0", "XYWAIT": "0", "HOME": "1", "ZENC": "1"}),
        ])
        dummy._read_hardware_position = lambda: next(positions)
        dummy.log_msg = lambda _message: None
        units, position = dummy._ensure_arduino_idle_for_manual()
        self.assertIn("STOP\n", fake.writes)
        np.testing.assert_array_equal(units, [100, 0])
        np.testing.assert_allclose(position, [1.0, 0.0, 20.0])

    def test_manual_motion_is_rejected_before_homing(self):
        dummy = object.__new__(PulseSerialMixin)
        dummy._read_hardware_position = lambda: (
            np.array([0, 0]), np.zeros(3),
            {"RUN": "0", "XYWAIT": "0", "HOME": "0", "ZENC": "1"},
        )
        with self.assertRaisesRegex(RuntimeError, "belum homing"):
            dummy._ensure_arduino_idle_for_manual()

    def test_status_parser_reads_equals_format(self):
        values = PulseSerialMixin._parse_status_line(
            "STATUS:RUN=0:XYRUN=0:ACTIVE=2:READY=0:X=2800:Y=1400:"
            "ZDECI=150:ZTGT=150:ZBUSY=0:ZENC=1:GRIP=1:SERVO=0"
        )
        self.assertEqual((values["X"], values["Y"]), ("2800", "1400"))
        self.assertEqual(values["GRIP"], "1")

    def test_position_memory_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            dummy = object.__new__(PulseSerialMixin)
            dummy.position_state_path = Path(directory) / "position.json"
            dummy.workspace_x_max = 580.0
            dummy.workspace_y_max = 500.0
            dummy.workspace_z_max = 300.0
            dummy.log_msg = lambda _message: None
            dummy._hardware_pos = np.array([10.0, 20.0, 30.0])
            dummy.current_pos = np.zeros(3)
            dummy._save_position_memory()
            dummy._hardware_pos = np.zeros(3)
            self.assertTrue(dummy._load_position_memory())
            np.testing.assert_allclose(dummy._hardware_pos, [10.0, 20.0, 30.0])
            np.testing.assert_allclose(dummy.current_pos, [10.0, 20.0, 30.0])

    def test_z_only_change_creates_angle_targets_without_xy_steps(self):
        points = np.array([[0, 0, 0], [0, 0, 100]], dtype=float)
        plan = plan_polyline(points, max_speed_mm_s=(80, 80, 20), z_deg_per_cm=15)
        self.assertTrue(np.all(plan.delta_steps == 0))
        self.assertEqual(plan.blocks[-1].slices[-1][2], 1500)

    def test_generated_route_preserves_action_order(self):
        route, action_map = DummyRoute()._route_from_generated_segments(False)
        actions = [action for index in sorted(action_map) for action in action_map[index]]
        self.assertEqual(actions, ["GRIP", "RELEASE", "GRIP", "RELEASE"])
        np.testing.assert_allclose(route[0], [0, 0, 0])
        np.testing.assert_allclose(route[-1], [70, 40, 20])

    def test_cycle_route_returns_to_grip(self):
        route, action_map = DummyRoute()._route_from_generated_segments(True)
        np.testing.assert_allclose(route[-1], [10, 10, 20])
        self.assertEqual(action_map[len(route) - 1], ["GRIP"])

    def test_run_log_is_saved_for_partial_stop_with_segment_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            dummy = object.__new__(PulseBlockExecutionMixin)
            dummy.run_log_dir = Path(directory)
            dummy.stop_requested = True
            dummy._start_run_log(False)
            dummy._append_run_log_segment(
                "LINE", np.zeros(3), np.array([10.0, 5.0, 2.0]), pass_label="awal"
            )
            path = dummy._save_run_log(False)
            self.assertTrue(path.exists())
            rows = path.read_text(encoding="utf-8").splitlines()
            self.assertIn("run_status", rows[0])
            self.assertIn("stopped_partial", rows[1])
            self.assertIn("queued_or_interrupted", rows[1])

    def test_combined_xyz_timeline_uses_xy_distance_and_mm_s(self):
        points = np.array([[0, 0, 0], [10, 0, 100]], dtype=float)
        plan = plan_polyline(points, max_speed_mm_s=100, slice_ms=10, block_ms=20)
        # Z 100 mm membatasi waktu. Karena HIGH dan LOW Z masing-masing minimal
        # 20 us, kapasitasnya 25 kHz atau 46,875 mm/s. Timeline konstan
        # dibulatkan ke slice 10 ms sehingga durasinya menjadi 2,14 s.
        self.assertAlmostEqual(plan.duration_s, 2.14)
        expected = np.rint(points[-1, :2] * plan.steps_per_mm).astype(int)
        np.testing.assert_array_equal(plan.delta_steps.sum(axis=0), expected)
        self.assertEqual(plan.delta_steps.shape[1], 2)
        self.assertEqual(plan.planned_mm[-1, 2], 100.0)

    def test_servo_command_contains_angle_sequence_and_crc(self):
        class FakeSerial:
            is_open = True

            def __init__(self):
                self.writes = []
                self.responses = [b"ACTION_ACK:1:GRIP:30\n", b"GOK\n"]

            def write(self, value):
                self.writes.append(value)

            def flush(self):
                return

            def readline(self):
                return self.responses.pop(0) if self.responses else b""

        dummy = object.__new__(PulseSerialMixin)
        dummy.ser = FakeSerial()
        dummy.stop_requested = False
        dummy.servo_action_sequence = 0
        dummy.log_msg = lambda _message: None
        self.assertTrue(dummy._send_grip_command("GRIP"))
        command = dummy.ser.writes[0].decode("ascii").strip()
        _servo, sequence, crc_text, action = command.split(":")
        self.assertEqual((sequence, action), ("1", "GRIP"))
        expected_crc = binascii.crc_hqx(b"GRIP", 0xFFFF)
        self.assertEqual(int(crc_text, 16), expected_crc)


if __name__ == "__main__":
    unittest.main()
