import unittest

import numpy as np

from scripts.diagnose_monodgp_m59h_geometry import (
    corners, geometry_rows, geometry_deltas, wrapped_degrees, filter_masks,
    serialize_rows, rounded_rows, compare_geometry,
)


def inputs():
    return {"calibration": np.array([[[700, 0, 620, -350], [0, 700, 180, 140], [0, 0, 1, 0]]], dtype=np.float32),
            "image_size": np.array([[1240, 360]], dtype=np.float32)}


def candidates():
    values = np.zeros((1, 50, 37), dtype=np.float32)
    values[..., 0] = 1  # Native Car -> product Vehicle; not class zero.
    values[..., 1] = .5
    values[..., 2:4] = .5
    values[..., 4:6] = .1
    values[..., 6] = 20
    values[..., 7] = 1  # Heading bin zero.
    values[..., 31:34] = [2, 2, 4]
    values[..., 34:36] = .5
    values[..., 36] = .5
    return values


class M59hGeometryTests(unittest.TestCase):
    def test_bottom_center_calibration_and_absolute_dimensions(self):
        values = candidates()
        unchanged = values.copy()
        rows, bins = geometry_rows(values, inputs())
        np.testing.assert_array_equal(values, unchanged)
        np.testing.assert_allclose(rows[0, 2:6], [558, 162, 682, 198], atol=2e-6)
        np.testing.assert_array_equal(rows[0, 6:9], [2, 2, 4])
        np.testing.assert_allclose(rows[0, 9:12], [.5, .8, 20], atol=1e-7)
        self.assertEqual(rows[0, 13], .25)
        self.assertEqual(bins[0], 0)

    def test_yaw_uses_2d_box_center_not_projected_3d_center(self):
        values = candidates()
        values[..., 34] = .8
        rows, _ = geometry_rows(values, inputs())
        self.assertEqual(rows[0, 12], 0)
        self.assertGreater(rows[0, 9], 1)

    def test_angular_difference_wraps_at_pi(self):
        self.assertAlmostEqual(float(wrapped_degrees(np.pi-1e-6, -np.pi+1e-6)), np.rad2deg(2e-6), places=8)

    def test_corners_bottom_center_and_yaw(self):
        rows, _ = geometry_rows(candidates(), inputs())
        box = corners(rows[:1])[0]
        np.testing.assert_allclose(box.max(axis=0), [2.5, .8, 21], atol=1e-7)
        np.testing.assert_allclose(box.min(axis=0), [-1.5, -1.2, 19], atol=1e-7)
        rows[:, 12] = np.pi / 2
        box = corners(rows[:1])[0]
        self.assertAlmostEqual(np.ptp(box[:, 0]), 2)
        self.assertAlmostEqual(np.ptp(box[:, 2]), 4)

    def test_heading_bin_change_not_hidden_by_small_raw_delta(self):
        reference, actual = candidates(), candidates()
        reference[..., 7:19] = 0
        actual[..., 7:19] = 0
        reference[..., 7] = 1e-6
        actual[..., 8] = 1e-6
        report, _, _, _ = compare_geometry(reference, actual, inputs(), np.arange(50), np.arange(50))
        self.assertEqual(report['heading_bin_changes'], 50)
        self.assertAlmostEqual(report['continuous_geometry']['yaw_degrees_delta']['max'], 30)

    def test_filter_before_confidence_and_after_text_are_distinct(self):
        values = candidates()
        values[..., 1] = .002
        values[..., 36] = .1
        rows, _ = geometry_rows(values, inputs())
        masks = filter_masks(values, rows)
        self.assertTrue(masks['native_score'].all())
        self.assertFalse(masks['nearby_score_before_text'].any())
        values[..., 1] = .004
        values[..., 36] = 1
        rows, _ = geometry_rows(values, inputs())
        masks = filter_masks(values, rows)
        self.assertTrue(masks['nearby_score_before_text'].all())
        self.assertFalse(masks['nearby_score_after_native_text'].any())

    def test_rounding_boundary_changes_text_without_large_continuous_error(self):
        ref, _ = geometry_rows(candidates(), inputs())
        actual = ref.copy()
        ref[:, 11] = 20.00499
        actual[:, 11] = 20.00501
        self.assertNotEqual(serialize_rows(ref)[0], serialize_rows(actual)[0])
        self.assertLess(geometry_deltas(ref, actual)['depth_m_delta'].max(), .0001)
        self.assertAlmostEqual(geometry_deltas(rounded_rows(ref), rounded_rows(actual))['depth_m_delta'].max(), .01)

    def test_identity_mismatch_fails_before_pairwise_measurement(self):
        with self.assertRaisesRegex(RuntimeError, 'Top-k identities'):
            compare_geometry(candidates(), candidates(), inputs(), np.arange(50), np.arange(50)[::-1])

    def test_zero_delta_and_excluded_class_are_explicit(self):
        values = candidates()
        values[..., 0] = 2
        report, _, _, _ = compare_geometry(values, values, inputs(), np.arange(50), np.arange(50))
        self.assertTrue(all(x['max'] == 0 for x in report['continuous_geometry'].values()))
        self.assertEqual(report['filter_decisions']['native_score']['reference_count'], 50)
        self.assertEqual(report['filter_decisions']['product_export']['reference_count'], 0)


if __name__ == '__main__':
    unittest.main()
