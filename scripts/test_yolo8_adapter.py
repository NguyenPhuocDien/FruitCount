"""Dependency-free tests for YOLO deployment settings, coverage and rounding."""
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'backend FruitCount' / 'yolo8'
spec = importlib.util.spec_from_file_location('yolo8_adapter_test', DIRECTORY / 'adapter.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((DIRECTORY / 'deploy_config.json').read_text(encoding='utf-8'))

    def test_real_config(self):
        c = adapter.settings(self.config)
        self.assertEqual(c['mode'], 'tiled_full')
        self.assertEqual(adapter.calibrated_count(10, c), 11)
        self.assertEqual(adapter.calibrated_count(0, c), 1)  # notebook intercept, not forced to zero

    def test_edges_and_small_images(self):
        self.assertEqual(adapter.tile_offsets(100, 100, 640, .25), [(0, 0)])
        self.assertEqual(adapter.tile_offsets(720, 1280, 640, .25),
                         [(0, 0), (80, 0), (0, 480), (80, 480), (0, 640), (80, 640)])

    def test_invalid_configs(self):
        for key, value in [('tile_overlap', 1), ('tile', 0), ('count_conf', .01),
                           ('calib_a', float('nan')), ('mode', 'unknown')]:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    adapter.settings(dict(self.config, **{key: value}))

    def test_inspection_without_ml_dependencies(self):
        result = adapter.predict(None, None, self.config, {'inspect': True})
        self.assertEqual(result['framework'], 'Ultralytics + WBF')


if __name__ == '__main__':
    unittest.main()
