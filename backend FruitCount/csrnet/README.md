# Chạy CSRNet++

## Vùng nghi bỏ sót

Upload label YOLO tương ứng ảnh, chạy CSRNet và chọn **Nghi bỏ sót** trong kết quả. Adapter trả thêm PNG grayscale `densityResponse`, mã hóa mật độ tương đối 0–255 theo đỉnh của ảnh (khác PNG heatmap màu). Frontend đo giá trị cực đại trong bbox label: thấp hơn ngưỡng đang chọn thì tô đỏ/viền vàng, còn lại viền xanh. Đây là công cụ kiểm tra phản hồi mật độ, không phải ghép từng prediction với ground truth và không xác định FN/recall. Ngưỡng mặc định 10% là ngưỡng trực quan tùy chỉnh, chưa được hiệu chuẩn trên validation set. Dữ liệu so sánh không thay đổi số đếm hay heatmap.

Thư mục này dùng adapter.py với csrnet_plus.py và checkpoint định dạng `{state, kw}` từ code huấn luyện đã cung cấp. Không tải backbone pretrained khi nạp model.

1. Cài Python và các gói `torch`, `torchvision`, `numpy`, `opencv-python-headless` vào cùng môi trường. Có thể dùng requirements.txt ở gốc dự án sau khi cài PyTorch phù hợp máy.
2. Chạy web, vào `/playground`, bấm Làm mới ở thư viện model máy chủ và chọn CSRNet FruitCount.
3. Chọn ảnh rồi chạy. Web hiển thị số đếm và density heatmap; không có bounding box.

Adapter sử dụng đường dẫn weights của model.json; trường weights trỏ tới Kaggle trong deploy_config.json không được dùng. Giữ nguyên mode/tile/tile_overlap và cấu hình kiến trúc của bản đánh giá.

Số đếm thô là tổng density chưa resize. Số hiển thị áp dụng `max(0, calib_a * raw_count + calib_b)` một lần. Source csrnet_plus.py không chứa bước calibration; cần đối chiếu công thức này với notebook đánh giá nếu notebook dùng quy ước khác. Heatmap chỉ chuẩn hóa màu để hiển thị, không dùng tính số đếm.

Nếu cấu hình kiến trúc khác checkpoint, adapter sẽ báo lỗi thay vì chạy với weight không khớp. Máy có CUDA khả dụng sẽ dùng GPU, nếu không dùng CPU. Chế độ tiled_full có thêm lượt chạy toàn ảnh nên ảnh rất lớn có thể cần nhiều RAM/VRAM.
