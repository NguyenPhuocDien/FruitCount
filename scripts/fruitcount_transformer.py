"""Full-image DEIMv2/ECDet inference matching the audited FruitCount report.

Architecture code lives with each operator-installed model. No pretrained
downloads, training code, or calibration are needed to load the final checkpoint.
"""
import copy
import importlib
import json
import math
from pathlib import Path


RUNTIMES = {
    'deimv2': ('deimv2_native', 'DEIM', 'DINOv3STAs'),
    'ecdet': ('ecdet_native', 'ECDet', 'ViTAdapter'),
}


def settings(config):
    """Keep the deployed protocol explicit instead of guessing from weights."""
    c = dict(config)
    if c.get('architecture') not in RUNTIMES:
        raise ValueError('architecture phải là deimv2 hoặc ecdet.')
    if c.get('mode') != 'full':
        raise ValueError('Adapter này dùng full-image như bảng kết quả được chọn.')
    for key in ('imgsz', 'max_det'):
        if isinstance(c.get(key), bool) or not isinstance(c.get(key), int) or c[key] <= 0:
            raise ValueError(f'{key} phải là số nguyên dương.')
    for key in ('min_score', 'det_conf', 'count_conf', 'nms_iou'):
        value = c.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{key} phải nằm trong [0, 1].')
    if c['min_score'] > min(c['det_conf'], c['count_conf']):
        raise ValueError('min_score không được lớn hơn ngưỡng hiển thị hoặc đếm.')
    if c.get('device', 'auto') not in ('auto', 'cpu', 'cuda:0'):
        raise ValueError('device phải là auto, cpu hoặc cuda:0.')
    if not isinstance(c.get('native_config'), str) or not c['native_config']:
        raise ValueError('Thiếu native_config.')
    return c


def model_file(model_dir, name):
    root = Path(model_dir).resolve()
    target = (root / name).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError(f'File model không hợp lệ hoặc bị thiếu: {name}')
    return target


def build_model(model_dir, weights_path, c, device):
    import torch

    package_name, detector, backbone = RUNTIMES[c['architecture']]
    runtime = importlib.import_module(package_name)
    native = json.loads(model_file(model_dir, c['native_config']).read_text(encoding='utf-8'))
    if native.get('model') != detector or native.get('num_classes') != 1:
        raise ValueError('Cấu hình kiến trúc phải khớp detector một lớp FruitCount.')
    if native.get(detector, {}).get('backbone') != backbone:
        raise ValueError('Backbone không khớp kiến trúc FruitCount đã chọn.')
    native['eval_spatial_size'] = [c['imgsz'], c['imgsz']]
    native[backbone]['weights_path'] = None
    if c['architecture'] == 'ecdet':
        native[backbone]['skip_load_backbone'] = True
    # Same schema/default/shared/injection mechanism as upstream YAMLConfig,
    # using the resolved training JSON; no YAML includes or dataset imports.
    registry = {name: dict(schema) for name, schema in runtime.GLOBAL_CONFIG.items()}
    for name, value in native.items():
        if name in registry and isinstance(value, dict):
            registry[name].update(copy.deepcopy(value))
        else:
            registry[name] = copy.deepcopy(value)
    model = runtime.create(detector, registry)
    postprocessor = runtime.create(native['postprocessor'], registry)
    checkpoint = torch.load(str(weights_path), map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('model'), dict):
        raise ValueError('Checkpoint phải chứa state_dict tại khóa model.')
    # best.pt already stores the selected EMA weights. Never substitute last.pt
    # or initialize missing tensors from a generic pretrained checkpoint.
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    model.encoder.eval_spatial_size = None
    model.decoder.eval_spatial_size = None
    return model.to(device).eval(), postprocessor.to(device).eval()


def predict_arrays(model, postprocessor, images, c, device):
    """BGR uint8 -> RGB/PIL resize -> ImageNet -> native postprocessor -> NMS."""
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms as T
    from torchvision.ops import nms

    transform = T.Compose([
        T.Resize((c['imgsz'], c['imgsz'])), T.ToTensor(),
        T.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    results = []
    with torch.inference_mode():
        for image in images:
            height, width = image.shape[:2]
            tensor = transform(Image.fromarray(np.ascontiguousarray(image[..., ::-1]))).unsqueeze(0).to(device)
            # Upstream scales XYXY by width, height, width, height.
            sizes = torch.tensor([[width, height]], device=device)
            with torch.autocast(device.type, enabled=device.type == 'cuda', dtype=torch.float16):
                outputs = model(tensor)
            outputs = {k: v.float() for k, v in outputs.items() if k in ('pred_logits', 'pred_boxes')}
            prediction = postprocessor(outputs, sizes)[0]
            boxes, scores = prediction['boxes'].float(), prediction['scores'].float()
            keep = (scores >= c['min_score']) & torch.isfinite(boxes).all(1) & torch.isfinite(scores)
            boxes, scores = boxes[keep], scores[keep]
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)
            keep = nms(boxes, scores, c['nms_iou'])[:c['max_det']]
            results.append((boxes[keep].cpu().numpy(), scores[keep].cpu().numpy()))
    return results


def format_result(boxes, scores, width, height, c, label='fruit'):
    """Return normalized web boxes and the fixed-threshold, uncalibrated count."""
    import numpy as np

    scores = np.asarray(scores, dtype=np.float32)
    visible = scores >= min(c['det_conf'], c['count_conf'])
    detections = []
    for box, score, keep in zip(boxes, scores, visible):
        if not keep:
            continue
        x1, y1, x2, y2 = (float(value) for value in box)
        detections.append({
            'id': f"{c['architecture']}-{len(detections)}", 'classId': 0,
            'label': label, 'confidence': float(score), 'color': '#2563eb',
            'x': x1 / width, 'y': y1 / height,
            'width': (x2 - x1) / width, 'height': (y2 - y1) / height,
        })
    count = int((scores >= c['count_conf']).sum())
    return {
        'detections': detections, 'predicted_count': count, 'count_raw': count, 'countCalibrated': False,
        'detectionThreshold': c['det_conf'], 'countThreshold': c['count_conf'],
        'modelClasses': [label], 'totalModelClasses': 1,
    }


def predict(image_path, weights_path, config, options):
    c = settings(config)
    architecture = 'DEIMv2-M' if c['architecture'] == 'deimv2' else 'ECDet-M O365'
    metadata = {
        'architecture': architecture, 'framework': 'PyTorch native',
        'inputShape': f"1 × 3 × {c['imgsz']} × {c['imgsz']}",
        'preprocess': {'imgsz': [c['imgsz'], c['imgsz']], 'norm': 'ImageNet',
                       'normKey': 'imagenet', 'channels': 3, 'layout': 'NCHW'},
        'notes': [
            f"Full-image {c['imgsz']}×{c['imgsz']}; RGB; ImageNet normalization; NMS IoU {c['nms_iou']}.",
            f"Đếm box có score ≥ {c['count_conf']}; không hiệu chỉnh tuyến tính.",
            'Ngưỡng hiển thị chỉ lọc box; số đếm dùng count_conf trong deploy_config.json.',
        ],
    }
    if options.get('inspect'):
        return metadata
    import cv2
    import torch

    requested = c.get('device', 'auto')
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Cấu hình yêu cầu CUDA nhưng môi trường Python không có CUDA khả dụng.')
    # Bound CPU thread oversubscription in the per-request Python process.
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Không đọc được ảnh đầu vào detector.')
    model, postprocessor = build_model(options['model_dir'], weights_path, c, device)
    boxes, scores = predict_arrays(model, postprocessor, [image], c, device)[0]
    height, width = image.shape[:2]
    classes = options.get('manifest', {}).get('classes') or ['fruit']
    return {**metadata, **format_result(boxes, scores, width, height, c, classes[0]), 'device': str(device)}
