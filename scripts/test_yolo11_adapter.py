"""Run with python scripts/test_yolo11_adapter.py (no ML dependencies needed)."""
import importlib.util
import json
import unittest
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parents[1] / 'backend FruitCount' / 'yolo11'
spec = importlib.util.spec_from_file_location('yolo11_adapter_test', DIRECTORY / 'adapter.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((DIRECTORY / 'deploy_config.json').read_text(encoding='utf-8'))

    def test_own_config_and_rounding(self):
        c = adapter.settings(self.config)
        self.assertEqual(c['arch'], 'yolo11m')
        self.assertEqual(c['count_conf'], .325)
        self.assertLess(c['count_conf'], c['det_conf'])
        self.assertEqual(adapter.calibrated_count(0, c), 0)
        self.assertEqual(adapter.calibrated_count(100, c), 101)

    def test_manifest_and_weight(self):
        manifest = json.loads((DIRECTORY / 'model.json').read_text(encoding='utf-8'))
        for key in ('weights', 'adapter', 'deployConfig'):
            self.assertTrue((DIRECTORY / manifest[key]).is_file())

    def test_inspection(self):
        result = adapter.predict(None, None, self.config, {'inspect': True})
        self.assertEqual(result['architecture'], 'yolo11m')


if __name__ == '__main__':
    unittest.main()
