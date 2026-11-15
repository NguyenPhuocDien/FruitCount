# YOLO11m FruitCount

Tích hợp theo notebook cv-fruitcount (1).ipynb. Chọn YOLO11m FruitCount trong thư viện model, thêm ảnh và ground truth nếu cần rồi chạy.

Các file cần có: best_fruitcount.pt, deploy_config.json, model.json và adapter.py. Kiến trúc được nạp từ checkpoint, không dựng lại theo p2. Đường dẫn Kaggle trong deploy_config.json không được dùng; weight local lấy từ model.json.

YOLO8 và YOLO11 dùng chung scripts/fruitcount_yolo.py vì các hàm raw/tiled_predict/fuse_wbf/predict_image và công thức AppleCounter trong hai notebook giống nhau. Mỗi thư mục đọc cấu hình riêng; không dùng hệ số của YOLO8 cho YOLO11.

Cấu hình hiện tại: M1_Photometric, YOLO11m không P2; full imgsz=832, tile=640, overlap=0.25, tile_imgsz=960; tiled_full. Raw conf=0.05, IoU=0.6, max_det=1500; bỏ box sát biên trong crop 2px, NMS tile IoU=0.5; WBF với toàn ảnh IoU=0.55, skip_box_thr=0.05, conf_type=avg.

Số box thô dùng count_conf=0.325. Số đếm hiệu chỉnh là round(max(1.0119043521267863 * count_raw - 0.5701510734932109, 0)). Ngưỡng hiển thị ban đầu det_conf=0.35, nên có box được tính vào số lượng nhưng không hiện trên ảnh. Hạ slider xuống 0.325 để xem chúng. Slider chỉ thay cách hiển thị, không thay count_conf hoặc tính lại calibration.

Dependency đã khai báo: torch/torchvision, ultralytics, numpy, opencv-python-headless, ensemble-boxes. CUDA dùng FP16/batch tile 16; CPU dùng FP32/batch tile 1. Chưa xác minh inference thật trong môi trường Codex không có Python; cần chạy trên máy có môi trường ML của bạn.
