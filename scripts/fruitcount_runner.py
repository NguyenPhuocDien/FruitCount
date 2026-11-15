"""Local, operator-installed model adapters. Never imports browser uploads."""
import importlib.util
import json
import math
import sys
import time
from pathlib import Path


def run_local(args):
    from run_inference import ModelError, handle_onnx, handle_torch, load_image_rgb, load_labels_file
    manifest = Path(args.manifest).resolve()
    root = manifest.parent
    config = json.loads(manifest.read_text(encoding='utf-8'))

    def local_file(name):
        target = (root / name).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ModelError(f'File không hợp lệ hoặc bị thiếu: {name}')
        return target

    args.model = str(local_file(config['weights']))
    deploy = json.loads(local_file(config['deployConfig']).read_text(encoding='utf-8')) if config.get('deployConfig') else {}
    labels = config.get('classes', [])
    if config.get('labelsFile'):
        labels = load_labels_file(str(local_file(config['labelsFile'])))
    base = {'modelName': config.get('name', root.name), 'sizeMB': round(Path(args.model).stat().st_size / 1024**2, 2)}
    if not config.get('adapter'):
        args.imgsz = config.get('imgsz', args.imgsz)
        args.norm = config.get('norm', args.norm)
        img = None if args.inspect else load_image_rgb(args.image)
        handler = handle_onnx if Path(args.model).suffix.lower() == '.onnx' else handle_torch
        return handler(args, img, labels, base)

    adapter = local_file(config['adapter'])
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('fruitcount_adapter', adapter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, 'predict', None)):
        raise ModelError('Adapter phải cung cấp hàm predict(image_path, weights_path, config, options).')
    started = time.perf_counter()
    result = module.predict(image_path=args.image, weights_path=args.model, config=deploy,
                            options={'conf': args.conf, 'iou': args.iou, 'manifest': config,
                                     'model_dir': str(root), 'inspect': args.inspect})
    if not isinstance(result, dict):
        raise ModelError('Adapter phải trả về dict kết quả.')
    if result.get('success') is False:
        raise ModelError(result.get('error', 'Adapter thất bại.'))
    if 'predicted_count' in result:
        count = float(result['predicted_count'])
        if not math.isfinite(count) or count < 0:
            raise ModelError('predicted_count phải là số hữu hạn không âm.')
        result['predicted_count'] = count
    if not args.inspect and config['family'] == 'csrnet' and 'predicted_count' not in result:
        raise ModelError('Adapter CSRNet chưa trả về predicted_count.')
    elapsed = (time.perf_counter() - started) * 1000
    return {**base, 'success': True, 'modelType': config['task'], 'task': config['task'],
            'detections': [], 'classes': labels, 'modelClasses': labels,
            'totalModelClasses': len(labels), 'classesCount': len(labels),
            'device': 'Python adapter', 'latencyMs': round(elapsed, 1),
            'fps': round(1000 / max(elapsed, .1), 1), 'timestamp': time.strftime('%H:%M:%S'), **result}
