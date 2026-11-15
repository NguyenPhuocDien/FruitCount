"""Bridge the supplied CSRNet++ checkpoint and Kaggle inference pipeline to AI Lab."""
import base64
import math


def _settings(config):
    mode = config.get('mode', 'full')
    if mode not in ('full', 'tta', 'tiled', 'tiled_full'):
        raise ValueError(f'Chế độ CSRNet không hợp lệ: {mode}')
    tile = config.get('tile', 512)
    if isinstance(tile, bool) or not isinstance(tile, int) or tile < 16 or tile % 8:
        raise ValueError('tile phải là số nguyên >= 16 và chia hết cho 8.')
    overlap = float(config.get('tile_overlap', 0.25))
    a, b = float(config.get('calib_a', 1)), float(config.get('calib_b', 0))
    if not math.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError('tile_overlap phải nằm trong [0, 1).')
    if not all(math.isfinite(x) for x in (a, b)):
        raise ValueError('Hệ số hiệu chỉnh phải là số hữu hạn.')
    return mode, tile, overlap, a, b


def predict(image_path, weights_path, config, options):
    mode, tile, overlap, a, b = _settings(config)
    if options.get('inspect'):
        return {'architecture': 'ImprovedCSRNet', 'framework': 'PyTorch',
                'inputShape': 'RGB, kích thước gốc; density stride 4',
                'notes': [f'Chế độ {mode}; tile={tile}, overlap={overlap}.']}

    import cv2
    import numpy as np
    import torch
    from csrnet_plus import ImprovedCSRNet, predict_density

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Không đọc được ảnh đầu vào CSRNet.')
    if min(image.shape[:2]) < 16:
        raise ValueError('Ảnh CSRNet cần chiều rộng và chiều cao ít nhất 16 pixel.')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # The provided training code saves {state, kw, epoch, val_MAE}.
    # Avoid its load_model's implicit pickle defaults, which differ across PyTorch versions.
    try:
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
    except TypeError:
        checkpoint = torch.load(weights_path, map_location='cpu')
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('state'), dict):
        raise ValueError('Checkpoint CSRNet phải chứa state (trọng số), như định dạng csrnet_plus.py lưu khi train.')
    architecture = checkpoint.get('kw')
    if architecture is None:
        architecture = {key: config[key] for key in ('use_ibn', 'use_se', 'aspp_rates') if key in config}
        if len(architecture) != 3:
            raise ValueError('Thiếu cấu hình kiến trúc kw trong checkpoint hoặc deploy_config.json.')
    if not isinstance(architecture, dict):
        raise ValueError('Checkpoint kw phải là dict cấu hình kiến trúc.')
    for key in ('use_ibn', 'use_se', 'aspp_rates'):
        if key in architecture and key in config:
            saved, deployed = architecture[key], config[key]
            if key == 'aspp_rates':
                saved, deployed = list(saved), list(deployed)
            if saved != deployed:
                raise ValueError(f'Cấu hình {key} không khớp giữa checkpoint và deploy_config.json.')
    model = ImprovedCSRNet(pretrained=False, **architecture)
    model.load_state_dict(checkpoint['state'], strict=True)
    del checkpoint
    model = model.to(device).eval()

    # Original pipeline expects BGR uint8 and performs ImageNet normalization internally.
    # Do not resize, normalize twice or divide the density sum by STRIDE squared.
    with torch.inference_mode():
        density = np.asarray(predict_density(model, image, mode, device,
                                            tile=tile, overlap=overlap), dtype=np.float32)
    if density.ndim != 2 or not density.size or not np.isfinite(density).all():
        raise ValueError('Density map không hợp lệ hoặc chứa NaN/Infinity.')
    raw_count = float(density.sum(dtype=np.float64))
    predicted_count = max(0.0, a * raw_count + b)

    # Normalize only a display copy. Count is computed on the original density grid.
    peak = float(density.max())
    scaled = np.clip(density / peak * 255, 0, 255).astype(np.uint8) if peak > 0 else np.zeros_like(density, dtype=np.uint8)
    heatmap = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    ok, png = cv2.imencode('.png', heatmap)
    if not ok:
        raise ValueError('Không mã hóa được heatmap CSRNet.')
    ok_response, response_png = cv2.imencode('.png', scaled)
    if not ok_response:
        raise ValueError('Không mã hóa được phản hồi mật độ CSRNet.')
    return {'predicted_count': predicted_count, 'raw_count': raw_count,
            'densityResponse': 'data:image/png;base64,' + base64.b64encode(response_png.tobytes()).decode('ascii'),
            'densityMap': 'data:image/png;base64,' + base64.b64encode(png.tobytes()).decode('ascii'),
            'detections': [], 'architecture': 'ImprovedCSRNet', 'framework': 'PyTorch',
            'device': str(device),
            'notes': [f'Inference: {mode}; tile={tile}; overlap={overlap}.',
                      f'Số đếm = max(0, {a} × tổng density + {b}).',
                      'CSRNet dự đoán mật độ, không tạo bounding box cho từng quả.']}
