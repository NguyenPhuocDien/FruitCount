# Chạy FruitCount AI bằng Docker

## Chuẩn bị

- Docker Desktop đang chạy ở chế độ **Linux containers**, hoặc Docker Engine + Compose v2 trên Linux.
- Máy có đủ RAM/dung lượng cho PyTorch, weight và ảnh. Image build lần đầu tải các thư viện ML; CPU suy luận tiled_full có thể chậm đáng kể.
- Bộ cấu hình PyTorch mặc định nhắm Linux x86_64. GPU NVIDIA cần driver hỗ trợ CUDA 12.4 và Docker GPU runtime (Docker Desktop WSL2 trên Windows hoặc NVIDIA Container Toolkit trên Linux).
- Giữ thư mục `backend FruitCount/` chứa weight, adapter và config. Không cần Node/Python cài trên máy chủ ngoài container.

## Chạy CPU

Tại thư mục chứa compose.yaml:

```powershell
docker compose up --build -d
docker compose ps
docker compose logs -f web
```

Mở http://localhost:3000/playground hoặc http://localhost:3000/compare. Đăng nhập demo vẫn hoạt động như bản local. Dừng bằng `docker compose down` (không xóa thư mục model).

## Triển khai cho thiết bị cùng Wi-Fi / LAN

```powershell
npm.cmd run deploy:lan
```

Lệnh build, khởi động container, đợi healthcheck và in các URL IPv4 LAN. Nếu image đã build: `npm.cmd run deploy:lan -- --no-build`. Mở URL dạng `http://192.168.x.x:3000/playground` trên thiết bị khác, không dùng localhost. Cùng ảnh/model vẫn xử lý trên máy chủ; không cần cài Python trên thiết bị truy cập.

Compose mặc định bind 0.0.0.0. Nếu chỉ muốn dùng một máy, đặt FRUITCOUNT_BIND_ADDRESS=127.0.0.1 và dùng docker compose up -d thông thường. deploy:lan chủ động override giá trị này thành 0.0.0.0.

Trên Windows, nếu bị Firewall chặn, mở PowerShell Administrator rồi chạy:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/allow-lan.ps1 -Port 3000
```

Quy tắc chỉ cho phép TCP trên mạng Private từ LocalSubnet, không mở toàn bộ Firewall. Chỉ đánh dấu mạng Private nếu đó là mạng tin cậy. Wi-Fi khách/AP isolation có thể chặn truy cập giữa các thiết bị dù server đã mở cổng. Để gỡ quy tắc: `Remove-NetFirewallRule -Name FruitCount-LAN-TCP-3000` trong PowerShell Administrator.

## Link HTTPS thử từ mạng khác

Khi image web đã build, chạy:

```powershell
npm.cmd run deploy:public
```

Lệnh khởi động Cloudflare Quick Tunnel và in URL `https://...trycloudflare.com/playground`, dùng được từ 4G/5G hoặc Wi-Fi khác. Không cần mở port router, tên miền hoặc tài khoản Cloudflare. Máy chủ, Docker và tunnel phải tiếp tục chạy; URL có thể đổi khi tunnel restart. Đây là demo công khai, chỉ chia sẻ cho người thử tin cậy; chưa thay thế triển khai production có xác thực server. Request inference quá lâu hoặc upload quá lớn có thể bị giới hạn bởi Cloudflare dù bản LAN chạy được.

Dừng chia sẻ Internet mà giữ web local:

```powershell
docker compose -f compose.yaml -f compose.tunnel.yaml stop tunnel
```

Với GPU, thêm `-f compose.gpu.yaml` vào lệnh Compose khi quản lý dịch vụ. Script deploy:public mặc định chạy image CPU.

Container chạy production server; `npm start` chỉ dành cho chạy local, không được dùng trong image. Không có npm/pip install lúc container khởi động. Healthcheck gọi `/api/health`; trạng thái healthy xác nhận web phản hồi, không bảo đảm từng weight đã load thành công.

## Cấu hình cổng và model

Không bắt buộc tạo .env. Nếu cần đổi mặc định, copy `.env.example` thành `.env`, ví dụ:

```dotenv
FRUITCOUNT_PORT=4000
FRUITCOUNT_BIND_ADDRESS=0.0.0.0
FRUITCOUNT_MODELS_DIR="D:/FruitCount/models"
INSTALL_TENSORFLOW=false
```

Thư mục host được mount vào `/models` **chỉ đọc**. Nên giữ nguyên cấu trúc:

```text
models/
  csrnet/    model.json, adapter.py, csrnet_plus.py, deploy_config.json, weight
  yolo8/     model.json, adapter.py, deploy_config.json, weight
  yolo11/    model.json, adapter.py, deploy_config.json, weight
  deimv2_m/  model.json, adapter.py, deploy_config.json, best.pt, native_model_config.json, deimv2_native/, source_files.json
  ecdet_m/   model.json, adapter.py, deploy_config.json, best.pt, native_model_config.json, ecdet_native/, source_files.json
```

Trong model.json, `weights`/`adapter`/`deployConfig` phải là đường dẫn tương đối bên trong thư mục của model; không dùng đường dẫn ổ D: hoặc Kaggle. Các thư mục mẫu chưa đủ file sẽ bị vô hiệu hóa trong danh sách, không chặn model đã cài.

Thêm/sửa model trên host rồi bấm Làm mới, không cần build image. Cache Python/ảnh upload tạm được ghi vào thư mục container riêng, không ghi vào weight. Bản sao chạy Docker không chia sẻ thư mục tạm với bản local.

## Chạy GPU NVIDIA

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
docker compose -f compose.yaml -f compose.gpu.yaml exec web python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Lệnh kiểm tra phải in True để adapter chọn CUDA. CPU và GPU dùng tên image khác nhau. Sau khi chuyển từ GPU về CPU, chạy lại `docker compose up --build -d` không có override. Dừng bản GPU bằng cùng hai cờ `-f` và lệnh `down`.

Dockerfile mặc định PyTorch 2.6.0 / torchvision 0.21.0; GPU override dùng wheel CUDA 12.4. Nếu checkpoint đòi phiên bản khác, cần đổi **cả cặp phiên bản tương thích** và index wheel qua build args TORCH_VERSION, TORCHVISION_VERSION, TORCH_INDEX_URL rồi build lại. Thư viện Python còn lại được giới hạn major version nhưng chưa có lockfile toàn bộ dependency; kiểm tra weight thực tế sau mỗi lần build lại.

Dockerfile nâng pip trước khi cài ML, tải riêng wheel torch/torchvision với `--no-deps` từ kho CPU/CUDA rồi cài dependency thông thường qua PyPI. Điều này tránh lỗi metadata `typing-extensions`/`typing_extensions` và thiếu `flit_core` khi resolver dùng kho PyTorch cho toàn bộ dependency.

## TensorFlow và model tùy biến

Image mặc định phục vụ YOLO, CSRNet, PyTorch/torchvision, ONNX. Để bật `.h5/.keras/.tflite`, đặt `INSTALL_TENSORFLOW=true` trong .env rồi chạy `docker compose build web` và `docker compose up -d`. TensorFlow là bản CPU, kể cả khi dùng GPU override cho PyTorch.

Hai adapter DEIMv2-M/M1 và ECDet-M/Original đã kèm source inference và dùng dependency PyTorch/torchvision hiện có; xem [pipeline transformer](TRANSFORMER_MODELS.md). DINO và các kiến trúc tùy biến khác vẫn cần source/adapter cùng dependency của chúng. Bộ Docker không tự sinh kiến trúc hoặc cài gói từ mã model lúc chạy.

## Mang sang máy khác, kể cả không có Internet lúc chạy

Sau khi build thành công:

```powershell
docker image save -o fruitcount-ai-cpu.tar fruitcount-ai:cpu
```

Copy image tar, compose.yaml và toàn bộ thư mục `backend FruitCount/` sang máy đích. Tại máy đích:

```powershell
docker image load -i fruitcount-ai-cpu.tar
docker compose up -d --no-build --pull never
```

Weight **không nằm trong image**, phải copy riêng; vẫn cần Docker trên máy đích. GPU tương tự nhưng dùng image fruitcount-ai:gpu và cả compose.gpu.yaml. Adapter/source thay đổi trong scripts/ của dự án cần build lại image, khác với các file trong thư mục model mount.

## Phạm vi triển khai

Compose mặc định mở cổng trên các interface máy chủ để dùng trong LAN. Tài khoản, mật khẩu và vai trò hiện là cơ chế demo trong localStorage; Docker không biến phần này thành xác thực server. Trước khi công khai Internet cần bổ sung xác thực API và giới hạn tài nguyên/upload phù hợp.

Docker Desktop phải được mở trước khi dùng lệnh up/build. Lỗi pipe dockerDesktopLinuxEngine/“Cannot connect to Docker daemon” nghĩa là engine chưa sẵn sàng, không phải lỗi model. Lần build đầu cần mạng tới registry, npm, PyPI, PyTorch và Google Fonts do layout đang dùng next/font/google.

## Kiểm tra sau build

Kiểm tra HTTP, danh sách model, giao diện và các file CSS/JS bằng Node trên máy host:

```powershell
node scripts/smoke-docker.mjs
```

Nếu đổi cổng: `node scripts/smoke-docker.mjs http://127.0.0.1:4000`. Script không chạy inference thật.

```powershell
docker compose exec web python -c "import torch, torchvision, ultralytics, ensemble_boxes; print(torch.__version__)"
```

Test scripts được loại khỏi image production. Chạy các test từ source bằng Python local: `python scripts/test_yolo8_adapter.py`, `python scripts/test_yolo11_adapter.py`. Kiểm tra inference thực tế bằng cùng ảnh/weight trên Playground, đối chiếu số box và số đếm sau hiệu chỉnh với notebook. Với mask1.png mẫu: 95 instance IDs; chế độ binary chỉ ra 70 vùng liên thông.

Tài liệu tham chiếu: [Next.js standalone](https://nextjs.org/docs/app/api-reference/config/next-config-js/output), [Compose GPU](https://docs.docker.com/compose/how-tos/gpu-support/), [cặp phiên bản PyTorch](https://docs.pytorch.org/get-started/previous-versions/).
