"""Shared YOLO8/YOLO11 FruitCount deployment from the supplied notebooks.

Single-class apple detector: raw YOLO -> tile NMS -> WBF -> count calibration.
"""
import math


def settings(config):
    c = dict(config)
    for key in ('imgsz', 'tile', 'tile_imgsz'):
        value = c.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{key} phải là số nguyên dương.')
    for key in ('det_conf', 'count_conf', 'tile_overlap'):
        c[key] = float(c[key])
        if not math.isfinite(c[key]) or not 0 <= c[key] <= 1:
            raise ValueError(f'{key} phải nằm trong [0, 1].')
    if c['tile_overlap'] >= 1:
        raise ValueError('tile_overlap phải nhỏ hơn 1.')
    if min(c['det_conf'], c['count_conf']) < 0.05:
        raise ValueError('Ngưỡng phải >= 0.05 vì pipeline notebook lọc raw ở 0.05.')
    for key in ('calib_a', 'calib_b'):
        c[key] = float(c.get(key, 1 if key == 'calib_a' else 0))
        if not math.isfinite(c[key]):
            raise ValueError(f'{key} phải hữu hạn.')
    if c.get('mode') not in ('full', 'tta', 'tiled', 'tiled_full'):
        raise ValueError('mode phải là full, tta, tiled hoặc tiled_full.')
    return c


def tile_offsets(width, height, tile, overlap):
    step = max(int(tile * (1 - overlap)), 1)
    xs = list(range(0, max(width - tile, 0) + 1, step))
    ys = list(range(0, max(height - tile, 0) + 1, step))
    if xs[-1] + tile < width:
        xs.append(max(width - tile, 0))
    if ys[-1] + tile < height:
        ys.append(max(height - tile, 0))
    return [(x, y) for y in ys for x in xs]


def calibrated_count(raw_count, c):
    return int(round(max(c['calib_a'] * raw_count + c['calib_b'], 0)))


def raw(model, images, imgsz, device, half):
    import numpy as np
    results = model.predict(images, imgsz=imgsz, conf=0.05, iou=0.6,
                            max_det=1500, verbose=False, half=half, device=device)
    return [(r.boxes.xyxy.cpu().numpy().astype(np.float32),
             r.boxes.conf.cpu().numpy().astype(np.float32)) for r in results]


def fuse(predictions, width, height):
    import numpy as np
    from ensemble_boxes import weighted_boxes_fusion
    if all(len(p[0]) == 0 for p in predictions):
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    scale = np.array([width, height, width, height], np.float32)
    boxes, scores, _ = weighted_boxes_fusion(
        [np.clip(p[0] / scale, 0, 1).tolist() for p in predictions],
        [p[1].tolist() for p in predictions], [[0] * len(p[1]) for p in predictions],
        iou_thr=0.55, skip_box_thr=0.05, conf_type='avg')
    return np.array(boxes, np.float32).reshape(-1, 4) * scale, np.array(scores, np.float32)


def tiled(model, image, c, device, half):
    import numpy as np
    import torch
    from torchvision.ops import nms
    height, width = image.shape[:2]
    offsets = tile_offsets(width, height, c['tile'], c['tile_overlap'])
    boxes, scores = [], []
    # Stream crop batches to bound memory. CUDA keeps the notebook's batch of 16.
    batch = 16 if half else 1
    for start in range(0, len(offsets), batch):
        part = offsets[start:start + batch]
        crops = [np.ascontiguousarray(image[y:y + c['tile'], x:x + c['tile']]) for x, y in part]
        predictions = raw(model, crops, c['tile_imgsz'], device, half)
        for (b, s), (x, y), crop in zip(predictions, part, crops):
            if len(b) == 0:
                continue
            h, w = crop.shape[:2]
            keep = np.ones(len(b), bool)
            if x > 0: keep &= b[:, 0] > 2
            if y > 0: keep &= b[:, 1] > 2
            if x + w < width: keep &= b[:, 2] < w - 2
            if y + h < height: keep &= b[:, 3] < h - 2
            b, s = b[keep], s[keep]
            if len(b):
                boxes.append(b + np.array([x, y, x, y], np.float32))
                scores.append(s)
    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    b, s = np.concatenate(boxes), np.concatenate(scores)
    indices = nms(torch.from_numpy(b), torch.from_numpy(s), 0.5).numpy()
    return b[indices], s[indices]


def infer(model, image, c, device, half):
    import numpy as np
    height, width = image.shape[:2]
    mode = c['mode']
    if mode == 'full':
        return raw(model, [image], c['imgsz'], device, half)[0]
    if mode == 'tta':
        predictions = [raw(model, [image], c['imgsz'], device, half)[0]]
        for axis, limit, columns in ((1, width, [0, 2]), (0, height, [1, 3])):
            b, s = raw(model, [np.ascontiguousarray(np.flip(image, axis=axis))], c['imgsz'], device, half)[0]
            b = b.copy()
            b[:, columns] = limit - b[:, columns[::-1]]
            predictions.append((b, s))
        return fuse(predictions, width, height)
    tiles = tiled(model, image, c, device, half)
    if mode == 'tiled':
        return tiles
    return fuse([tiles, raw(model, [image], c['imgsz'], device, half)[0]], width, height)


def predict(image_path, weights_path, config, options):
    c = settings(config)
    if options.get('inspect'):
        return {'architecture': c.get('arch', 'YOLOv8'), 'framework': 'Ultralytics + WBF',
                'inputShape': f"{c['imgsz']}×{c['imgsz']}; tiles {c['tile_imgsz']}"}
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    try:
        import ensemble_boxes  # fail early with an actionable message
    except ImportError as exc:
        raise ValueError('Thiếu ensemble-boxes. Cài vào môi trường Python chạy web: pip install ensemble-boxes') from exc
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Không đọc được ảnh đầu vào YOLO.')
    model = YOLO(weights_path)
    names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    if len(names) != 1 or model.task != 'detect':
        raise ValueError('Adapter notebook FruitCount chỉ dành cho detector một lớp quả.')
    half = torch.cuda.is_available()
    device = 0 if half else 'cpu'
    with torch.inference_mode():
        boxes, scores = infer(model, image, c, device, half)
    height, width = image.shape[:2]
    raw_count = int((scores >= c['count_conf']).sum())
    detections = []
    # Keep all boxes used by either threshold; det_conf controls initial display.
    for b, score in zip(boxes, scores):
        if float(score) < min(c['det_conf'], c['count_conf']):
            continue
        x1, y1, x2, y2 = np.clip(b, [0, 0, 0, 0], [width, height, width, height])
        detections.append({'id': f'yolo-{len(detections)}', 'classId': 0, 'label': names[0],
                           'confidence': float(score), 'color': '#2563eb',
                           'x': float(x1 / width), 'y': float(y1 / height),
                           'width': float((x2 - x1) / width), 'height': float((y2 - y1) / height)})
    return {'detections': detections, 'predicted_count': calibrated_count(raw_count, c),
            'count_raw': raw_count, 'detectionThreshold': c['det_conf'], 'countThreshold': c['count_conf'],
            'modelClasses': names, 'totalModelClasses': 1, 'device': 'cuda:0' if half else 'cpu',
            'framework': 'Ultralytics YOLO + WBF', 'architecture': c.get('arch', 'YOLOv8'),
            'notes': [f"{c['mode']}: YOLO IoU 0.6; tile NMS 0.5; WBF IoU 0.55.",
                      f"Đếm {raw_count} box với score ≥ {c['count_conf']}; hiệu chỉnh và làm tròn: max(0, {c['calib_a']} × count + {c['calib_b']}).",
                      'Ngưỡng trên giao diện chỉ lọc box hiển thị; số đếm hiệu chỉnh dùng count_conf trong deploy_config.json.']}
