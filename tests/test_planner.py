import sys
import unittest
import binascii
import importlib.util
from pathlib import Path

import numpy as np

_planner_path = Path(__file__).resolve().parents[1] / "python_app" / "pulse_planner.py"
_planner_spec = importlib.util.spec_from_file_location("python_app_pulse_planner", _planner_path)
_planner_module = importlib.util.module_from_spec(_planner_spec)
assert _planner_spec.loader is not None
sys.modules[_planner_spec.name] = _planner_module
_planner_spec.loader.exec_module(_planner_module)
plan_polyline = _planner_module.plan_polyline
plan_sinusoidal = _planner_module.plan_sinusoidal


class PlannerTest(unittest.TestCase):
    def test_crc16_ccitt_reference_vector(self):
        self.assertEqual(binascii.crc_hqx(b"123456789", 0xFFFF), 0x29B1)

    def test_sine_returns_to_origin(self):
        plan = plan_sinusoidal(axis="X", amplitude_mm=20, period_s=4, cycles=2)
        self.assertEqual(int(plan.delta_steps[:, 0].sum()), 0)
        self.assertTrue(np.all(plan.delta_steps[:, 1:] == 0))
        self.assertEqual([b.sequence for b in plan.blocks], list(range(len(plan.blocks))))

    def test_polyline_final_steps_match_target(self):
        points = np.array([[0, 0, 0], [50, 20, 5], [100, 30, 10]], dtype=float)
        plan = plan_polyline(points, max_speed_mm_s=60)
        accumulated = plan.delta_steps.sum(axis=0)
        expected = np.rint(points[-1, :2] * plan.steps_per_mm).astype(int)
        np.testing.assert_array_equal(accumulated, expected)

    def test_block_size_limit(self):
        plan = plan_sinusoidal(slice_ms=10, block_ms=20)
        self.assertTrue(all(len(block.slices) <= 2 for block in plan.blocks))
        self.assertTrue(all(len(block.command.encode("ascii")) < 128 for block in plan.blocks))

    def test_xy_mm_s_fits_20us_timer_with_2us_step_pulse(self):
        steps_per_cm = (2800.0, 1400.0)
        x_speed = 80.0
        points = np.array([[0, 0, 0], [100, 0, 0]], dtype=float)
        plan = plan_polyline(points, max_speed_mm_s=x_speed, steps_per_cm=steps_per_cm)
        self.assertLessEqual(int(np.abs(plan.delta_steps[:, 0]).max()), 500)
        self.assertLessEqual(float(np.abs(plan.pulse_rate[:, 0]).max()), 50000.0)

    def test_block_payload_contains_xy_and_z_angle_target(self):
        plan = plan_sinusoidal(axis="Y", amplitude_mm=5, period_s=1, cycles=1)
        self.assertTrue(all(len(step_target) == 3 for block in plan.blocks for step_target in block.slices))
        self.assertTrue(all(step_target[2] == 0 for block in plan.blocks for step_target in block.slices))

    def test_unlimited_speed_input_is_accepted_and_timer_safe(self):
        points = np.array([[0, 0, 0], [500, 0, 0]], dtype=float)
        plan = plan_polyline(points, max_speed_mm_s=100000.0)
        self.assertLessEqual(int(np.abs(plan.delta_steps).max()), 500)
        self.assertGreater(len(plan.blocks), 0)

    def test_polyline_default_has_no_start_or_end_ramp(self):
        points = np.array([[0, 0, 0], [100, 0, 0]], dtype=float)
        plan = plan_polyline(points, max_speed_mm_s=80.0)
        x_steps = np.abs(plan.delta_steps[:, 0])
        self.assertGreater(int(x_steps[0]), 0)
        self.assertGreater(int(x_steps[-1]), 0)
        self.assertLessEqual(int(x_steps.max() - x_steps.min()), 1)
        self.assertAlmostEqual(plan.duration_s, 1.25, places=2)

    def test_legacy_constant_speed_flag_does_not_change_timeline(self):
        points = np.array([[0, 0, 0], [100, 20, 10]], dtype=float)
        default_plan = plan_polyline(points, max_speed_mm_s=(80, 60, 20))
        legacy_plan = plan_polyline(
            points, max_speed_mm_s=(80, 60, 20), constant_speed=True
        )
        np.testing.assert_array_equal(default_plan.times_s, legacy_plan.times_s)
        np.testing.assert_array_equal(default_plan.delta_steps, legacy_plan.delta_steps)

    def test_different_xy_speed_uses_slowest_required_axis_time(self):
        points = np.array([[0, 0, 0], [100, 100, 0]], dtype=float)
        plan = plan_polyline(
            points,
            max_speed_mm_s=(100.0, 50.0),
            constant_speed=True,
        )
        # X memerlukan 1 s, Y memerlukan 2 s; keduanya selesai bersama 2 s.
        self.assertAlmostEqual(plan.duration_s, 2.0, places=2)
        np.testing.assert_array_equal(
            plan.delta_steps.sum(axis=0),
            np.rint(points[-1, :2] * plan.steps_per_mm).astype(int),
        )

    def test_z_target_uses_15_degree_per_cm(self):
        points = np.array([[0, 0, 0], [0, 0, 10]], dtype=float)
        plan = plan_polyline(
            points,
            max_speed_mm_s=(80.0, 80.0, 10.0),
            z_deg_per_cm=15.0,
            constant_speed=True,
        )
        # 10 mm = 1 cm = 15° = 150 deci-degree.
        self.assertEqual(plan.blocks[-1].slices[-1][2], 150)

    def test_z_targets_stay_synchronized_with_plot_segments(self):
        points = np.array([[0, 0, 0], [100, 0, 0], [100, 0, 100]], dtype=float)
        plan = plan_polyline(
            points,
            max_speed_mm_s=(80.0, 80.0, 20.0),
            z_deg_per_cm=15.0,
        )
        z_targets = np.asarray(
            [0] + [target for block in plan.blocks for _, _, target in block.slices]
        )
        x_deltas = np.asarray(
            [dx for block in plan.blocks for dx, _, _ in block.slices]
        )
        x_targets = np.concatenate(([0], np.cumsum(x_deltas)))
        first_z_motion = int(np.flatnonzero(z_targets != 0)[0])
        # Selama jalur masih berada pada segmen Z=0, X harus sudah mencapai
        # titik belok sebelum target Z mulai berubah.
        self.assertGreaterEqual(x_targets[first_z_motion], 100 * plan.steps_per_mm[0])
        self.assertEqual(z_targets[0], 0)
        self.assertEqual(z_targets[-1], 1500)

    def test_completed_block_can_be_released_and_rebuilt(self):
        points = np.array([[0, 0, 0], [100, 0, 10]], dtype=float)
        plan = plan_polyline(points, max_speed_mm_s=60)
        original = plan.get_block(0)
        plan.release_block(0)
        self.assertIsNone(plan.blocks[0])
        rebuilt = plan.get_block(0)
        self.assertEqual(rebuilt.command, original.command)

    def test_long_sequence_numbers_are_not_limited_to_uint16(self):
        from pulse_planner import PulseBlock

        block = PulseBlock(70000, ((1, 0, 0),))
        self.assertTrue(block.command.startswith("BLOCK:70000:1:"))


if __name__ == "__main__":
    unittest.main()
