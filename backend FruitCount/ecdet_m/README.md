# ECDet-M O365 — Original

Model số 4 trong báo cáo `FRUITCOUNT_RESULTS_REPORT_2026-11-12.md`.

- Checkpoint: `best.pt` của `fruitcount_ecdet/runs/Original`, chứa EMA state tại khóa `model`.
- Train set: Original, 500 ảnh MinneApple; O365 chỉ nguồn pretraining của detector.
- Full-image 832×832, RGB + ImageNet normalization, NMS IoU 0.6.
- Ngưỡng hiển thị 0.35; ngưỡng đếm 0.40; **không hiệu chỉnh tuyến tính**.
- Tham chiếu trên 170 ảnh validation: MAE 3.841176, mAP50 0.873146. Không dùng kết quả tiled_full mAP50 0.885401 để gắn cho mode full này.

Source kiến trúc và license được giữ trong `ecdet_native/`; config kiến trúc đầy đủ ở `native_model_config.json`. `source_files.json` ghi nguồn và SHA256 từng file. Runtime không tải backbone/pretrained từ Internet; checkpoint phải load strict toàn bộ.

Chọn **ECDet-M O365 — Original** tại `/playground` hoặc `/compare`. Xem [hướng dẫn chung](../../docs/TRANSFORMER_MODELS.md).
