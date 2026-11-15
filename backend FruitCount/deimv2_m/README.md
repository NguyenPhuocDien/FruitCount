# DEIMv2-M — M1 Photometric

Model số 1 trong báo cáo `FRUITCOUNT_RESULTS_REPORT_2026-11-12.md`.

- Checkpoint: `best.pt` của `fruitcount_deimv2/runs/M1`, chứa EMA state tại khóa `model`.
- Train set: M1 Photometric; không phải Original hoặc Original_merged.
- Full-image 832×832, RGB + ImageNet normalization, NMS IoU 0.6.
- Ngưỡng hiển thị 0.35; ngưỡng đếm 0.40; **không hiệu chỉnh tuyến tính**.
- Tham chiếu trên 170 ảnh validation: MAE 3.352941, mAP50 0.879123. Đây là kết quả lịch sử, không phải metric tính lại khi cài web.

Source kiến trúc và license được giữ trong `deimv2_native/`; config kiến trúc đầy đủ ở `native_model_config.json`. `source_files.json` ghi nguồn và SHA256 từng file. Runtime không tải backbone/pretrained từ Internet; checkpoint phải load strict toàn bộ.

Chọn **DEIMv2-M — M1 Photometric** tại `/playground` hoặc `/compare`. Xem [hướng dẫn chung](../../docs/TRANSFORMER_MODELS.md).
