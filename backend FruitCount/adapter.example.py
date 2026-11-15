"""Copy into a model directory as adapter.py and implement its original pipeline."""


def predict(image_path, weights_path, config, options):
    raise NotImplementedError(
        'Cần tích hợp source/load_model/predict của model đã huấn luyện vào adapter.py. '
        'Xem backend FruitCount/README.md; không thể suy ra kiến trúc tùy biến chỉ từ weight.'
    )
