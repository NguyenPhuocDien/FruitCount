# YOLOv8m FruitCount

Adapter dựa trên các hàm raw, tiled_predict, fuse_wbf, predict_image và AppleCounter của notebook `fruitcount-sota (1).ipynb` do người dùng cung cấp. Weight được nạp từ model.json, không dùng đường dẫn Kaggle trong deploy_config.json.

Pipeline:

- Raw YOLO: conf=0.05, IoU=0.6, max_det=1500.
- Toàn ảnh: imgsz trong deploy_config.json.
- Tile: kích thước tile, tile_overlap và tile_imgsz từ config; bỏ box chạm mép trong của crop trong phạm vi 2 pixel; dịch về tọa độ ảnh gốc và NMS IoU=0.5.
- tiled_full: WBF giữa dự đoán tile và toàn ảnh, IoU=0.55, skip_box_thr=0.05, conf_type=avg; tương tự notebook, toàn bộ box thuộc một lớp.
- count_raw: số box sau gộp có score >= count_conf.
- predicted_count: int(round(max(calib_a * count_raw + calib_b, 0))). Đây là số đếm hiệu chỉnh, có thể khác số box nhìn thấy.
- det_conf khởi tạo ngưỡng hiển thị trên web. Thay slider chỉ đổi box hiển thị; không tái hiệu chuẩn số đếm.

Không lấy conf/IoU trên giao diện để thay các hằng số hậu xử lý notebook. Muốn đổi pipeline, sửa deploy_config.json rồi chạy lại. P2 nằm trong kiến trúc checkpoint; adapter không dựng lại head theo cờ p2.

Cần torch, torchvision, ultralytics, numpy, opencv-python-headless và ensemble-boxes trong môi trường Python chạy web. Launcher npm start đã bổ sung kiểm tra ensemble-boxes. Nếu chạy npm run dev trực tiếp, cài bằng `pip install ensemble-boxes` với cùng Python/PYTHON_BIN của backend.

CUDA dùng FP16 và batch tile=16 giống notebook. CPU dùng FP32 và batch tile=1 để chạy được với ít bộ nhớ hơn; có thể có sai lệch nhỏ do độ chính xác số học. Chưa kiểm thử inference thật trong môi trường Codex vì không có Python khả dụng; cần đối chiếu với notebook trên cùng ảnh và weight.
