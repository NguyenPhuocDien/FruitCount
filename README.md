# 🧪 AI Lab – Computer Vision Playground

## Chạy bằng Docker

Để thiết bị khác cùng Wi-Fi/LAN truy cập, chạy `npm.cmd run deploy:lan`. Lệnh sẽ triển khai Docker và in URL IP để mở trên điện thoại/máy khác. Hướng dẫn Windows Firewall nằm trong [DOCKER.md](docs/DOCKER.md).

Mở Docker Desktop (Linux containers), giữ đầy đủ weight/source/config trong `backend FruitCount/`, rồi chạy:

```powershell
docker compose up --build -d
```

Mở http://localhost:3000/playground hoặc http://localhost:3000/compare. Image CPU là mặc định; weight mount riêng, không đóng vào image. Có cấu hình GPU NVIDIA và tùy chọn TensorFlow. Xem [hướng dẫn Docker, chuyển máy và cấu hình](docs/DOCKER.md).

## Model FruitCount trên máy chủ

Đặt từng model trong thư mục `backend FruitCount/`, cấu hình `model.json`, rồi chọn tại Playground → **Thư viện FruitCount trên máy chủ**. Có adapter cho YOLO, CSRNet, **DEIMv2-M — M1 Photometric** và **ECDet-M O365 — Original**; ResNet và DINO còn là cấu hình mẫu. Xem [hướng dẫn cài model và adapter](backend%20FruitCount/README.md) và [pipeline DEIMv2/ECDet](docs/TRANSFORMER_MODELS.md). Các kiến trúc tùy biến cần source và adapter tương ứng; không thể chạy chỉ bằng file trọng số.

Website kiểm thử mô hình Computer Vision: tải mô hình (`.pt`, `.onnx`, `.h5`, `.keras`, `.tflite`, YOLO…) và ảnh lên để chạy inference trực tiếp.

## ✅ Yêu cầu (chỉ cài 1 lần)

| Phần mềm             | Phiên bản                  | Tải về                                                                         |
| -------------------- | -------------------------- | ------------------------------------------------------------------------------ |
| **Node.js**          | ≥ 20 (khuyên dùng bản LTS) | https://nodejs.org                                                             |
| **Python**           | 3.10 – 3.12                | https://www.python.org/downloads/ (Windows: tick **"Add python.exe to PATH"**) |
| **Git** _(tùy chọn)_ | bất kỳ                     | https://git-scm.com                                                            |

## 🚀 Chạy web – 1 lệnh

```bash
git clone https://github.com/HoangMinnn/Website_CV.git
cd Website_CV
npm start
```

> Không có Git? Bấm **Code → Download ZIP** trên GitHub, giải nén, rồi:
>
> - **Windows:** nhấp đúp file **`start.bat`**
> - **macOS / Linux:** chạy `./start.sh`

Lệnh này sẽ **tự động**:

1. Kiểm tra Node.js và Python
2. Cài các thư viện Python còn thiếu (PyTorch, Ultralytics YOLO, ONNX Runtime, TensorFlow…)
3. Cài thư viện Node.js (`npm install`)
4. Build và khởi động web, **tự mở trình duyệt** tại http://localhost:3000

⏱ Lần đầu mất khoảng **5–10 phút** (tải PyTorch/TensorFlow). Các lần sau chỉ vài giây.

**Tài khoản demo**

| Vai trò | Email             | Mật khẩu   |
| ------- | ----------------- | ---------- |
| Admin   | `admin@ailab.com` | `admin123` |
| User    | `user@ailab.com`  | `user123`  |

## ⚙️ Tùy chọn

| Lệnh                                                         | Tác dụng                                                             |
| ------------------------------------------------------------ | -------------------------------------------------------------------- |
| `npm start -- --dev`                                         | Chạy chế độ phát triển (tự reload khi sửa code)                      |
| `npm start -- --no-open` | Không tự mở trình duyệt                                  |
| `PORT=4000 npm start`    | Đổi cổng (mặc định 3000, tự chọn cổng khác nếu bị chiếm) |

## 🌐 Triển khai online lên Render.com

Dự án đã có sẵn file `render.yaml`. Các bước triển khai:

1. Đăng nhập [Render.com](https://render.com) bằng tài khoản GitHub của bạn.
2. Nhấn nút **New +** ở góc trên → chọn **Web Service**.
3. Chọn kết nối với kho lưu trữ **`Website_CV`**.
4. Điền các thông số (hoặc Render sẽ tự nhận qua `render.yaml`):
   - **Name:** `website-cv` (tuỳ ý)
   - **Runtime:** `Node`
   - **Build Command:**
     ```bash
     npm install && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision && pip install --no-cache-dir -r requirements.txt && npm run build
     ```
   - **Start Command:**
     ```bash
     npm run serve
     ```
   - **Instance Type:** Chọn gói **Free**
   - Mở mục **Advanced** → thêm biến môi trường (Environment Variable):
     - `NODE_VERSION` = `20`
     - `NODE_OPTIONS` = `--max-old-space-size=400`
5. Bấm **Deploy Web Service** và chờ Render hoàn tất build!

## 🧱 Công nghệ

- **Frontend:** Next.js 16, React 19, Tailwind CSS 4
- **Inference:** Python – PyTorch, Ultralytics YOLO, ONNX Runtime, TensorFlow/Keras

## ❓ Lỗi thường gặp

- **`Không tìm thấy Python`** → cài Python và nhớ tick _Add to PATH_, rồi mở lại terminal.
- **Cài TensorFlow thất bại** → web vẫn chạy bình thường, chỉ không mở được mô hình `.h5/.keras/.tflite`. Thường do dùng Python 3.13+; hãy dùng Python 3.12.
- **Có GPU NVIDIA nhưng chạy bằng CPU** → cài PyTorch bản CUDA theo hướng dẫn tại https://pytorch.org/get-started/locally/
