"""Tests for transformer deployment, counts, paths and integrity; NumPy only."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from fruitcount_transformer import format_result, model_file, predict, settings


ROOT = Path(__file__).resolve().parents[1]


class TransformerTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'backend FruitCount/deimv2_m/deploy_config.json').read_text())

    def test_fixed_protocol_and_distinct_architectures(self):
        for model_id, architecture, dataset in [('deimv2_m', 'deimv2', 'M1 Photometric'), ('ecdet_m', 'ecdet', 'Original')]:
            with self.subTest(model=model_id):
                c = settings(json.loads((ROOT / 'backend FruitCount' / model_id / 'deploy_config.json').read_text()))
                self.assertEqual((c['architecture'], c['dataset']), (architecture, dataset))
                self.assertEqual((c['imgsz'], c['mode'], c['count_conf']), (832, 'full', .4))

    def test_count_threshold_is_separate_from_display_and_not_calibrated(self):
        c = settings(self.config)
        result = format_result([[10, 20, 50, 80]] * 4, [.34, .35, .399, .4], 100, 200, c)
        self.assertEqual(len(result['detections']), 3)
        self.assertEqual(result['predicted_count'], 1)
        self.assertEqual(result['count_raw'], 1)
        self.assertFalse(result['countCalibrated'])
        self.assertEqual(result['detections'][0]['x'], .1)
        self.assertEqual(result['detections'][0]['y'], .1)
        self.assertEqual(result['detections'][0]['width'], .4)
        self.assertEqual(result['detections'][0]['height'], .3)
        self.assertEqual(format_result([], [], 100, 200, c)['predicted_count'], 0)

    def test_invalid_config(self):
        for key, value in [('architecture', 'yolo'), ('mode', 'tiled_full'), ('imgsz', True),
                           ('imgsz', 0), ('max_det', -1), ('det_conf', float('nan')),
                           ('count_conf', 1.1), ('min_score', .5), ('device', 'gpu'), ('native_config', '')]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                settings(dict(self.config, **{key: value}))

    def test_inspect_does_not_require_ml_or_weights(self):
        result = predict(None, None, self.config, {'inspect': True})
        self.assertEqual(result['architecture'], 'DEIMv2-M')
        self.assertEqual(result['preprocess']['normKey'], 'imagenet')

    def test_files_cannot_escape_model_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'model'
            root.mkdir()
            (Path(folder) / 'outside.json').write_text('{}')
            with self.assertRaises(ValueError):
                model_file(root, '../outside.json')

    def test_installed_files_match_packaged_hashes_and_manifests(self):
        for model_id in ('deimv2_m', 'ecdet_m'):
            root = ROOT / 'backend FruitCount' / model_id
            manifest = json.loads((root / 'model.json').read_text(encoding='utf-8'))
            inventory = json.loads((root / 'source_files.json').read_text(encoding='utf-8'))
            for name in [manifest['weights'], manifest['adapter'], manifest['deployConfig'], *manifest['requiredFiles']]:
                self.assertTrue((root / name).is_file(), name)
            for record in inventory['files']:
                with (root / record['path']).open('rb') as stream:
                    self.assertEqual(hashlib.file_digest(stream, 'sha256').hexdigest(), record['sha256'], record['path'])


if __name__ == '__main__':
    unittest.main()
