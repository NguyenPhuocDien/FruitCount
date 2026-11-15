# Thư viện model FruitCount

Mỗi thư mục con là một model, có `model.json`. Có thể sao chép thư mục mẫu để thêm phiên bản. Tên thư mục chỉ dùng chữ không dấu, số, `_`, `-`.

Chạy `npm run dev`, mở `/playground`, chọn ở **Thư viện FruitCount trên máy chủ**. Sau khi thêm file, bấm **Làm mới**. Không cần upload lại weight mỗi lần chạy. Các đường dẫn trong model.json tương đối với thư mục model. Không sửa package-lock.json để thêm model.

## Các thư mục model

| Thư mục | Cần bổ sung | Runner |
| --- | --- | --- |
| yolo8, yolo11 | Weight, adapter và config đã có trên máy | Ultralytics + pipeline tile/WBF riêng |
| resnet | best.pth và classes đúng thứ tự | TorchScript, nn.Module hoặc state_dict torchvision chuẩn |
| csrnet | best_fruitcount_csrnet.pth, deploy_config.json, csrnet_plus.py, adapter.py và module phụ thuộc | Adapter riêng |
| dino | best.pth, deploy_config.json, adapter.py và source/config gốc | Adapter riêng |
| deimv2_m | Đã cài best.pt, native config, source inference và adapter | DEIMv2-M / M1 Photometric, full-image |
| ecdet_m | Đã cài best.pt, native config, source inference và adapter | ECDet-M O365 / Original, full-image |

Tên weight tùy ý, sửa `weights` trong model.json. `labelsFile` tùy chọn trỏ đến file nhãn txt/json/yaml. YOLO dùng nhãn nhúng trong model nếu có. ResNet phân loại không tự trở thành model đếm quả; cần đúng kiến trúc/head và quy trình đã huấn luyện. ResNet regression hoặc kiến trúc tùy biến cần adapter.

ResNet và DINO vẫn là mẫu cấu hình cần bổ sung file. Trạng thái có đủ file không bảo đảm checkpoint tương thích. Hai model transformer đã được strict-load và smoke inference; xem [pipeline và nguồn checkpoint](../docs/TRANSFORMER_MODELS.md). Weight bị gitignore nên khi chuyển máy cần copy riêng.

## Adapter cho model tùy biến

Thêm `adapter.py` cạnh model.json, dựa trên `adapter.example.py` ở thư mục cha. Đây là mã Python do người vận hành cài sẵn, không phải file tải lên từ trình duyệt. Hàm bắt buộc:

```python
def predict(image_path, weights_path, config, options):
    # config: nội dung deploy_config.json, giữ nguyên mọi trường.
    # options: conf, iou, manifest, model_dir, inspect.
    # Import kiến trúc gốc, load weight, tiền xử lý và dự đoán thật tại đây.
    return {"detections": [], "predicted_count": 12.5}
```

Với CSRNet, dùng đúng hàm load_model/predict_density trong csrnet_plus.py. Phải đối chiếu chữ ký hàm thực tế trước khi viết adapter; dự án không giả định chữ ký của file chưa được cung cấp. Áp dụng mode, tile, tile_overlap, calib_a, calib_b, use_ibn, use_se, aspp_rates đúng pipeline Kaggle. Không tự cộng density sau resize nếu chưa bảo toàn tổng mật độ; không hiệu chỉnh hai lần. Trả predicted_count hữu hạn không âm, tùy chọn densityMap là data URL PNG của heatmap. segmentation chỉ là nhóm hiển thị; density map không phải mặt nạ phân lớp.

Với DINO và DEIMv2-M, cần source đúng phiên bản và cấu hình kiến trúc/checkpoint từ repo huấn luyện; hậu xử lý đúng scale ảnh, nhãn, threshold, NMS nếu pipeline yêu cầu. Tên DINO có thể chỉ detector DINO hoặc backbone DINO/DINOv2: adapter phải ứng với model thực tế. Không dùng một kiến trúc giả định chung cho các weight này.

Kết quả detection: `detections` là danh sách `{id, label, confidence, color, x, y, width, height}`. Tọa độ chuẩn hóa 0–1 theo ảnh gốc, confidence 0–1, color dạng CSS (ví dụ #22c55e). Trả `predicted_count` nếu pipeline có quy tắc đếm riêng. Phân loại trả `topClasses: [{label, confidence}]`. Có thể bổ sung device, notes, framework, architecture. Nếu lỗi, raise Exception hoặc trả `{success: False, error: "..."}`; web hiển thị lỗi, không thay bằng dự đoán mô phỏng.

## Môi trường Python

Cài PyTorch/torchvision phù hợp máy và `pip install -r requirements.txt` tại gốc repo. Các model tùy biến có thể cần dependency riêng từ repo huấn luyện; cài vào cùng môi trường Python. Đặt PYTHON_BIN nếu cần chọn Python khác. Các file deploy_config.json phải lấy từ lần huấn luyện/đánh giá thực tế, không dùng giá trị phỏng đoán.

Weight được bỏ qua bởi .gitignore. Giữ mã nguồn, config và hướng dẫn để tái lập. Mỗi request hiện nạp lại model trong tiến trình Python riêng; chưa có cache GPU hoặc hàng đợi.
