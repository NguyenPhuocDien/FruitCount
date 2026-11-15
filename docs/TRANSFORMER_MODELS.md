# Hai model transformer FruitCount

## Model được cài

| ID thư mục | Model / bộ train | Mode mặc định | MAE tham chiếu | mAP50 tham chiếu |
|---|---|---|---:|---:|
| `deimv2_m` | DEIMv2-M / M1 Photometric | full 832 | 3.352941 | 0.879123 |
| `ecdet_m` | ECDet-M O365 / Original | full 832 | 3.841176 | 0.873146 |

Hai dòng tương ứng hạng 1 và 4 của bảng full-image trong báo cáo ngày 12/11/2026. Metric dùng 170 ảnh validation, không phải test độc lập; đây là kết quả báo cáo gốc, chưa chạy lại toàn bộ tập validation trong web.

Nguồn checkpoint:

- [DEIMv2-M/M1, notebook part 2](https://www.kaggle.com/code/vyxwolf/fruitcount-deimv2-m-ddp-m1-v2-20261111-part-2): `fruitcount_deimv2/runs/M1/best.pt`.
- [ECDet-M O365/Original](https://www.kaggle.com/code/vyxwolf/fruitcount-ecdet-m-o365-ddp-20261111): `fruitcount_ecdet/runs/Original/best.pt`.

Các liên kết chỉ để truy nguyên artifact; adapter không truy cập Kaggle lúc chạy. Dùng đúng `best.pt`, không thay bằng pretrained O365/COCO hoặc `last.pt`. Checkpoint không phải Ultralytics YOLO.

## Cách dùng

1. Giữ nguyên toàn bộ thư mục `backend FruitCount/deimv2_m` và `backend FruitCount/ecdet_m`, bao gồm source package, checkpoint, manifest, config và license.
2. Chạy web bằng quy trình hiện có (`npm start` hoặc Docker Compose), mở `/playground` hoặc `/compare` rồi làm mới thư viện model.
3. Chọn tên model, tải ảnh, chạy inference. Compare tự dùng cùng ảnh cho cả hai model và chạy lần lượt.

Weight `.pt` vẫn bị gitignore như các model trước; khi gửi project cho đồng đội phải copy weight riêng. Source, config và README có thể đi cùng repository. `requiredFiles` liệt kê đầy đủ source để thư viện phát hiện bộ cài bị thiếu file.

## Pipeline giữ đúng báo cáo

```text
Ảnh BGR (OpenCV)
→ RGB (PIL)
→ Resize vuông 832×832, không letterbox
→ ToTensor + ImageNet mean/std
→ Model native + PostProcessor (width, height của ảnh gốc)
→ Lọc score >= 0.05, finite, clip box và NMS IoU 0.6, max_det 1500
→ Số đếm = số box có score >= 0.40
→ Box chuẩn hóa 0–1 cho ResultsPanel
```

Ngưỡng hiển thị ban đầu 0.35. Slider chỉ lọc box trả về, không thay ngưỡng đếm cố định. Không dùng WBF, tile hay hiệu chỉnh tuyến tính của pipeline YOLO. Backend trả `countCalibrated: false` để giao diện không ghi nhầm số đếm là đã hiệu chỉnh.

`deploy_config.json` chọn `device: auto`: CUDA khi môi trường PyTorch hỗ trợ, CPU khi không có. Có thể đặt `cpu` hoặc `cuda:0` rõ ràng trong config. Các lựa chọn khác bị từ chối. Không dùng lựa chọn device chưa được nối của giao diện Playground để cấu hình hai adapter này.

## Cấu trúc code

- `scripts/fruitcount_transformer.py`: kiểm tra cấu hình, dựng model native, strict-load checkpoint, preprocessing, inference, NMS và chuyển JSON dùng chung.
- `adapter.py` trong từng model: entrypoint mỏng theo hợp đồng hiện có.
- `deimv2_native/`, `ecdet_native/`: chỉ source phục vụ inference, namespace tách biệt. Các file toán học của upstream được copy nguyên vẹn; các `__init__.py` local chỉ đăng ký module cần thiết, bỏ dependency training/dataset. ECDet có helper distributed query tối thiểu để không import dataloader.
- `native_model_config.json`: bản cấu hình đã resolve từ lượt train; runner không thực thi include YAML hoặc mở đường dẫn dataset trong metadata.
- `source_files.json`: upstream commit, notebook nguồn và SHA256 file đã đóng gói; license upstream được giữ cạnh source.

Không thêm dependency ML ngoài nhóm NumPy/Pillow/OpenCV/PyTorch/torchvision đã có. Không nạp pretrained bổ sung, không tự cài package khi chạy. Docker hiện có copy `scripts/*.py` và mount thư viện model nên giữ được cùng luồng. Khi cập nhật `scripts/fruitcount_transformer.py` cần build lại image; chỉ thay file trong thư viện mount thì không cần build lại.

## Kiểm tra

```powershell
python scripts/test_transformer_adapter.py
```

Test cấu hình, phân biệt ngưỡng hiển thị/đếm, tọa độ ảnh không vuông, trường hợp không có detection, path traversal, manifest và hash file. Cần NumPy cho phần định dạng kết quả; các test này không load checkpoint.

Kiểm tra inference thực qua entrypoint web:

```powershell
python scripts/run_inference.py --manifest "backend FruitCount/deimv2_m/model.json" --model "backend FruitCount/deimv2_m/best.pt" --image "path/to/image.jpg"
python scripts/run_inference.py --manifest "backend FruitCount/ecdet_m/model.json" --model "backend FruitCount/ecdet_m/best.pt" --image "path/to/image.jpg"
```

Nếu dùng Python trong venv, đặt `PYTHON_BIN` cho server theo tài liệu hiện có. Bằng chứng kiểm tra lần tích hợp nằm trong `analysis_outputs/model_integration/`: parity CPU với Predictor gốc và smoke request HTTP production. Một ảnh smoke không chứng minh metric toàn dataset.
