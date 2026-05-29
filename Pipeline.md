# Hướng dẫn: Triển khai FastTD3 trên HumanoidBench với Modal (A100)

Tài liệu này ghi chú lại toàn bộ pipeline chuẩn xác để thiết lập, khắc phục lỗi và huấn luyện thuật toán **FastTD3** trên môi trường **HumanoidBench**, sử dụng nền tảng đám mây **Modal (NVIDIA A100 GPU)**.

---

## Bước 1: Tạo file cấu hình Modal (nếu chưa có) (`modal_jupyter.py`)

Trên máy tính cá nhân, tạo một file tên là `modal_jupyter.py` với nội dung dưới đây. Script này sẽ tự động cài đặt PyTorch (CUDA 12.1), các thư viện hệ thống cần thiết, và tải mã nguồn trực tiếp từ nhánh `main` để tránh lỗi thắt cổ chai của HumanoidBench.

```python
import modal
import subprocess

app = modal.App("fasttd3-a100-workspace")

# 1. TẠO Ổ CỨNG VĨNH CỬU: Nó sẽ tự động tạo nếu chưa có
model_storage = modal.Volume.from_name("fasttd3-models-storage", create_if_missing=True)

# Cấu hình Image theo chuẩn (đã fix sẵn EGL)
fasttd3_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "libglfw3", "libgl1-mesa-glx", "libosmesa6", 
        "git-lfs", "cmake", "xvfb", "ffmpeg", "git",
        "libegl1", "libegl1-mesa-dev"
    )
    .pip_install(
        "jupyterlab", "stable-baselines3", "wandb", "tensorboard", "pyvirtualdisplay"
    )
    .run_commands(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "pip install --editable git+https://github.com/carlosferrazza/humanoid-bench.git#egg=humanoid-bench",
        "git clone https://github.com/younggyoseo/FastTD3.git /workspace/FastTD3",
        "cd /workspace/FastTD3 && pip install -r requirements/requirements.txt"
    )
)

# 2. GẮN Ổ CỨNG VÀO ĐƯỜNG DẪN CỦA MÁY ẢO
@app.function(
    image=fasttd3_image, 
    gpu="A100", 
    cpu=16, 
    timeout=86400, # Giới hạn chạy tối đa 24 tiếng
    volumes={"/workspace/FastTD3/models": model_storage} # <--- Điểm mấu chốt ở đây!
)
def run_jupyter():
    import modal
    print("Đang khởi tạo đường hầm kết nối và gắn ổ cứng vĩnh cửu...")
    
    with modal.forward(8888) as tunnel:
        print(f"\n🚀 TRUY CẬP JUPYTER LAB TẠI LINK SAU: {tunnel.url}\n")
        
        subprocess.run([
            "jupyter", "lab", 
            "--ip=0.0.0.0", 
            "--port=8888", 
            "--allow-root", 
            "--no-browser", 
            "--notebook-dir=/workspace/FastTD3",
            "--ServerApp.token=''",
            "--ServerApp.password=''"
        ])
```

### 📌 Tính năng mới: Ổ cứng vĩnh cửu (Persistent Storage)

Phiên bản cập nhật của `modal_jupyter.py` bao gồm:

1. **Tạo Ổ cứng vĩnh cửu:** `modal.Volume.from_name("fasttd3-models-storage", create_if_missing=True)`
   - Tự động tạo một ổ cứng trên Modal nếu chưa tồn tại
   - Ổ cứng này sẽ **lưu trữ các model được huấn luyện** ngay cả khi session kết thúc

2. **Gắn ổ cứng vào container:** `volumes={"/workspace/FastTD3/models": model_storage}`
   - Tất cả các model lưu trong thư mục `/workspace/FastTD3/models` sẽ tự động được lưu vĩnh viễn
   - Lần chạy tiếp theo, bạn có thể tải lại các model đã huấn luyện trước đó

**Lợi ích:** Không mất mô hình khi session kết thúc hoặc timeout!

---

## Bước 2: Khởi chạy Jupyter Lab trên Đám mây

1. Mở Terminal trên máy tính cá nhân.

2. Di chuyển đến thư mục chứa file `modal_jupyter.py`.

3. Chạy lệnh:

```bash
modal run modal_jupyter.py
```

4. Chờ hệ thống Build Image và Provisioning (khoảng vài phút). Khi terminal in ra dòng `🚀 TRUY CẬP JUPYTER LAB TẠI LINK SAU: https://...`, hãy nhấp vào đường link đó để mở giao diện Jupyter Lab.

---

## Bước 3: Fix lỗi EGL & Đăng nhập Weights & Biases (Trong Jupyter Lab)

Bên trong giao diện Jupyter Lab, mở một Terminal mới (**File > New > Terminal**) và thực hiện các bước sau:

### 3.1 Di chuyển vào thư mục làm việc:

```bash
cd /workspace/FastTD3
```

### 3.2 Khắc phục lỗi EGL của MuJoCo:

Cập nhật và cài đặt thư viện EGL

```bash
apt-get update && apt-get install -y libegl1 libegl1-mesa-dev
```

Cài đặt biến môi trường ép MuJoCo sử dụng driver EGL cho phần cứng A100.

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

### 3.3 Xác thực Weights & Biases (W&B) để vẽ biểu đồ:

```bash
wandb login
```

(Dán API Key từ tài khoản wandb.ai của bạn vào đây).

---

## Bước 4: Khởi chạy Huấn Luyện (Training)

### 4.1 Giải thích Chi tiết các Tham số

Lệnh training cơ bản có cấu trúc:

```bash
python fast_td3/train.py \
    --env-name [TÊN_MÔI_TRƯỜNG] \
    --exp-name [TÊN_THỰC_NGHIỆM] \
    --seed [SỐ_NGẪU_NHIÊN] \
    --project [TÊN_DỰ_ÁN_WANDB] \
    --render-interval [BƯỚC_LƯU] \
    --compile-mode [CHẾ_ĐỘ_BIÊN_DỊCh]
```

**Giải thích từng tham số:**

| Tham số | Ý nghĩa | Ví dụ |
|--------|---------|-------|
| `--env-name` | **Tên môi trường HumanoidBench cần huấn luyện**. Xem danh sách đầy đủ ở mục 4.2 | `h1hand-hurdle-v0` |
| `--exp-name` | **Tên thực nghiệm** để phân biệt các lần chạy khác nhau trên W&B | `FastTD3_PPO_Reward` |
| `--seed` | **Số hạt ngẫu nhiên** để tái tạo kết quả. Dùng seed khác nhau để chạy thí nghiệm khác nhau | `1`, `2`, `3` |
| `--project` | **Tên dự án W&B** - tất cả exp trong dự án được nhóm lại để so sánh | `FastTD3_HumanoidBench_Experiments` |
| `--render-interval` | **Bao nhiêu bước huấn luyện mới lưu 1 video demo** (số step) | `5000` |
| `--compile-mode` | **Chế độ biên dịch Triton** - `max-autotune` tối ưu hóa tốc độ tối đa | `max-autotune` |

### 4.2 Danh sách Environments & Ví dụ Huấn Luyện

#### 1️⃣ Nhóm Di chuyển & Thăng bằng (Locomotion & Balance)

**Đi bộ (Walk):**
```bash
python fast_td3/train.py \
    --env-name h1hand-walk-v0 \
    --exp-name FastTD3_Walk_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune
```

**Chạy (Run):**
```bash
python fast_td3/train.py \
    --env-name h1hand-run-v0 \
    --exp-name FastTD3_Run_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune
```

**Nhảy vượt rào (Hurdle):**
```bash
python fast_td3/train.py \
    --env-name h1hand-hurdle-v0 \
    --exp-name FastTD3_Hurdle_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune
```

**Giữ thăng bằng (Balance):**
```bash
python fast_td3/train.py \
    --env-name h1hand-balance-v0 \
    --exp-name FastTD3_Balance_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune
```

**Tìm đường trong mê cung (Maze):**
```bash
python fast_td3/train.py \
    --env-name h1hand-maze-v0 \
    --exp-name FastTD3_Maze_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune
```

**Đi thăng bằng trên gờ hẹp (Pole):**
```bash
python fast_td3/train.py \
    --env-name h1hand-pole-v0 \
    --exp-name FastTD3_Pole_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune
```

---

#### 2️⃣ Nhóm Thao tác cơ bản (Basic Manipulation)

**Chạm mục tiêu (Reach):**
```bash
python fast_td3/train.py \
    --env-name h1hand-reach-v0 \
    --exp-name FastTD3_Reach_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune
```

**Đẩy khối hộp (Push):**
```bash
python fast_td3/train.py \
    --env-name h1hand-push-v0 \
    --exp-name FastTD3_Push_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune
```

**Xoay khối lập phương (Cube):**
```bash
python fast_td3/train.py \
    --env-name h1hand-cube-v0 \
    --exp-name FastTD3_Cube_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune
```

**Cắm chốt vào lỗ (Insert):**
```bash
python fast_td3/train.py \
    --env-name h1hand-insert-v0 \
    --exp-name FastTD3_Insert_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune
```

---

#### 3️⃣ Nhóm Phối hợp toàn thân (Whole-body / Real-world)

**Ném bóng rổ (Basketball):**
```bash
python fast_td3/train.py \
    --env-name h1hand-basketball-v0 \
    --exp-name FastTD3_Basketball_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune
```

**Đu xà đơn (Highbar):**
```bash
python fast_td3/train.py \
    --env-name h1hand-highbar-v0 \
    --exp-name FastTD3_Highbar_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune
```

**Cử tạ (Powerlift):**
```bash
python fast_td3/train.py \
    --env-name h1hand-powerlift-v0 \
    --exp-name FastTD3_Powerlift_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune
```

**Mở tủ (Cabinet):**
```bash
python fast_td3/train.py \
    --env-name h1hand-cabinet-v0 \
    --exp-name FastTD3_Cabinet_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune
```

**Mở cửa ra vào (Door):**
```bash
python fast_td3/train.py \
    --env-name h1hand-door-v0 \
    --exp-name FastTD3_Door_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune
```

**Mở cửa sổ (Window):**
```bash
python fast_td3/train.py \
    --env-name h1hand-window-v0 \
    --exp-name FastTD3_Window_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune
```

**Sắp xếp giá sách (Bookshelf):**
```bash
python fast_td3/train.py \
    --env-name h1hand-bookshelf-v0 \
    --exp-name FastTD3_Bookshelf_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune
```

**Tương tác nhà bếp (Kitchen):**
```bash
python fast_td3/train.py \
    --env-name h1hand-kitchen-v0 \
    --exp-name FastTD3_Kitchen_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune
```

**Dùng thìa (Spoon):**
```bash
python fast_td3/train.py \
    --env-name h1hand-spoon-v0 \
    --exp-name FastTD3_Spoon_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune
```

**Bê thùng hàng (Package):**
```bash
python fast_td3/train.py \
    --env-name h1hand-package-v0 \
    --exp-name FastTD3_Package_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune
```

**Chất hàng lên xe tải (Truck):**
```bash
python fast_td3/train.py \
    --env-name h1hand-truck-v0 \
    --exp-name FastTD3_Truck_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune
```

**Dọn dẹp căn phòng (Room):**
```bash
python fast_td3/train.py \
    --env-name h1hand-room-v0 \
    --exp-name FastTD3_Room_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune
```

---

### 💡 Ghi chú khi quan sát Log:

- **Cảnh báo Gym/NumPy 2.0:** Bạn có thể bỏ qua các cảnh báo màu đỏ liên quan đến "Gym has been unmaintained". Quá trình khởi tạo 128 môi trường song song sẽ mất khoảng 1-2 phút.

- **Autotune Triton:** Các dòng dạng `AUTOTUNE mm(32768x216, 216x1024)` xuất hiện ở khoảng 100 step đầu tiên là tính năng JIT compiler đang tự động tối ưu hóa mạng nơ-ron trên chip A100. Khi quá trình này xong, FPS huấn luyện sẽ tăng vọt.

- **Biểu đồ:** Click vào đường link W&B hiện ra trong Terminal để theo dõi điểm số theo thời gian thực.

---

### 4.3 Thực nghiệm SimbaV2 (Agent Architecture)

**SimbaV2** là một kiến trúc neural network tối ưu hóa với các tham số đặc biệt giúp cải thiện hiệu suất trên các task phức tạp. Danh sách lệnh dưới đây sử dụng agent `fasttd3_simbav2` với các siêu tham số được tinh chỉnh:

**Tham số SimbaV2 chính:**
- `--agent fasttd3_simbav2` - Sử dụng kiến trúc SimbaV2
- `--batch-size 8192` - Kích thước batch lớn hơn cho hiệu suất tốt hơn
- `--critic-learning-rate-end 3e-5` - Learning rate cho Critic network
- `--actor-learning-rate-end 3e-5` - Learning rate cho Actor network
- `--weight-decay 0.0` - Không áp dụng weight decay
- `--critic-hidden-dim 512` - Số chiều hidden của Critic
- `--critic-num-blocks 2` - Số block của Critic
- `--actor-hidden-dim 256` - Số chiều hidden của Actor
- `--actor-num-blocks 1` - Số block của Actor

#### 1️⃣ SimbaV2: Nhóm Di chuyển & Thăng bằng

**Đi bộ (Walk) - SimbaV2:**
```bash
python fast_td3/train.py \
    --env-name h1hand-walk-v0 \
    --exp-name FastTD3_SimbaV2_Walk_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune \
    --agent fasttd3_simbav2 \
    --batch-size 8192 \
    --critic-learning-rate-end 3e-5 \
    --actor-learning-rate-end 3e-5 \
    --weight-decay 0.0 \
    --critic-hidden-dim 512 \
    --critic-num-blocks 2 \
    --actor-hidden-dim 256 \
    --actor-num-blocks 1
```

**Chạy (Run) - SimbaV2:**
```bash
python fast_td3/train.py \
    --env-name h1hand-run-v0 \
    --exp-name FastTD3_SimbaV2_Run_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune \
    --agent fasttd3_simbav2 \
    --batch-size 8192 \
    --critic-learning-rate-end 3e-5 \
    --actor-learning-rate-end 3e-5 \
    --weight-decay 0.0 \
    --critic-hidden-dim 512 \
    --critic-num-blocks 2 \
    --actor-hidden-dim 256 \
    --actor-num-blocks 1
```

**Nhảy vượt rào (Hurdle) - SimbaV2:**
```bash
python fast_td3/train.py \
    --env-name h1hand-hurdle-v0 \
    --exp-name FastTD3_SimbaV2_Hurdle_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune \
    --agent fasttd3_simbav2 \
    --batch-size 8192 \
    --critic-learning-rate-end 3e-5 \
    --actor-learning-rate-end 3e-5 \
    --weight-decay 0.0 \
    --critic-hidden-dim 512 \
    --critic-num-blocks 2 \
    --actor-hidden-dim 256 \
    --actor-num-blocks 1
```

**Giữ thăng bằng (Balance) - SimbaV2:**
```bash
python fast_td3/train.py \
    --env-name h1hand-balance-v0 \
    --exp-name FastTD3_SimbaV2_Balance_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune \
    --agent fasttd3_simbav2 \
    --batch-size 8192 \
    --critic-learning-rate-end 3e-5 \
    --actor-learning-rate-end 3e-5 \
    --weight-decay 0.0 \
    --critic-hidden-dim 512 \
    --critic-num-blocks 2 \
    --actor-hidden-dim 256 \
    --actor-num-blocks 1
```

**Tìm đường trong mê cung (Maze) - SimbaV2:**
```bash
python fast_td3/train.py \
    --env-name h1hand-maze-v0 \
    --exp-name FastTD3_SimbaV2_Maze_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune \
    --agent fasttd3_simbav2 \
    --batch-size 8192 \
    --critic-learning-rate-end 3e-5 \
    --actor-learning-rate-end 3e-5 \
    --weight-decay 0.0 \
    --critic-hidden-dim 512 \
    --critic-num-blocks 2 \
    --actor-hidden-dim 256 \
    --actor-num-blocks 1
```

**Đi thăng bằng trên gờ hẹp (Pole) - SimbaV2:**
```bash
python fast_td3/train.py \
    --env-name h1hand-pole-v0 \
    --exp-name FastTD3_SimbaV2_Pole_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune \
    --agent fasttd3_simbav2 \
    --batch-size 8192 \
    --critic-learning-rate-end 3e-5 \
    --actor-learning-rate-end 3e-5 \
    --weight-decay 0.0 \
    --critic-hidden-dim 512 \
    --critic-num-blocks 2 \
    --actor-hidden-dim 256 \
    --actor-num-blocks 1
```

---

#### 2️⃣ SimbaV2: Nhóm Thao tác cơ bản

**Chạm mục tiêu (Reach) - SimbaV2:**
```bash
python fast_td3/train.py \
    --env-name h1hand-reach-v0 \
    --exp-name FastTD3_SimbaV2_Reach_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune \
    --agent fasttd3_simbav2 \
    --batch-size 8192 \
    --critic-learning-rate-end 3e-5 \
    --actor-learning-rate-end 3e-5 \
    --weight-decay 0.0 \
    --critic-hidden-dim 512 \
    --critic-num-blocks 2 \
    --actor-hidden-dim 256 \
    --actor-num-blocks 1
```

**Đẩy khối hộp (Push) - SimbaV2:**
```bash
python fast_td3/train.py \
    --env-name h1hand-push-v0 \
    --exp-name FastTD3_SimbaV2_Push_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune \
    --agent fasttd3_simbav2 \
    --batch-size 8192 \
    --critic-learning-rate-end 3e-5 \
    --actor-learning-rate-end 3e-5 \
    --weight-decay 0.0 \
    --critic-hidden-dim 512 \
    --critic-num-blocks 2 \
    --actor-hidden-dim 256 \
    --actor-num-blocks 1
```

**Xoay khối lập phương (Cube) - SimbaV2:**
```bash
python fast_td3/train.py \
    --env-name h1hand-cube-v0 \
    --exp-name FastTD3_SimbaV2_Cube_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune \
    --agent fasttd3_simbav2 \
    --batch-size 8192 \
    --critic-learning-rate-end 3e-5 \
    --actor-learning-rate-end 3e-5 \
    --weight-decay 0.0 \
    --critic-hidden-dim 512 \
    --critic-num-blocks 2 \
    --actor-hidden-dim 256 \
    --actor-num-blocks 1
```

**Cắm chốt vào lỗ (Insert) - SimbaV2:**
```bash
python fast_td3/train.py \
    --env-name h1hand-insert-v0 \
    --exp-name FastTD3_SimbaV2_Insert_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune \
    --agent fasttd3_simbav2 \
    --batch-size 8192 \
    --critic-learning-rate-end 3e-5 \
    --actor-learning-rate-end 3e-5 \
    --weight-decay 0.0 \
    --critic-hidden-dim 512 \
    --critic-num-blocks 2 \
    --actor-hidden-dim 256 \
    --actor-num-blocks 1
```

---

#### 3️⃣ SimbaV2: Nhóm Phối hợp toàn thân (Real-world)

**Ném bóng rổ (Basketball) - SimbaV2:**
```bash
python fast_td3/train.py \
    --env-name h1hand-basketball-v0 \
    --exp-name FastTD3_SimbaV2_Basketball_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune \
    --agent fasttd3_simbav2 \
    --batch-size 8192 \
    --critic-learning-rate-end 3e-5 \
    --actor-learning-rate-end 3e-5 \
    --weight-decay 0.0 \
    --critic-hidden-dim 512 \
    --critic-num-blocks 2 \
    --actor-hidden-dim 256 \
    --actor-num-blocks 1
```

**Đu xà đơn (Highbar) - SimbaV2:**
```bash
python fast_td3/train.py \
    --env-name h1hand-highbar-v0 \
    --exp-name FastTD3_SimbaV2_Highbar_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune \
    --agent fasttd3_simbav2 \
    --batch-size 8192 \
    --critic-learning-rate-end 3e-5 \
    --actor-learning-rate-end 3e-5 \
    --weight-decay 0.0 \
    --critic-hidden-dim 512 \
    --critic-num-blocks 2 \
    --actor-hidden-dim 256 \
    --actor-num-blocks 1
```

**Cử tạ (Powerlift) - SimbaV2:**
```bash
python fast_td3/train.py \
    --env-name h1hand-powerlift-v0 \
    --exp-name FastTD3_SimbaV2_Powerlift_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune \
    --agent fasttd3_simbav2 \
    --batch-size 8192 \
    --critic-learning-rate-end 3e-5 \
    --actor-learning-rate-end 3e-5 \
    --weight-decay 0.0 \
    --critic-hidden-dim 512 \
    --critic-num-blocks 2 \
    --actor-hidden-dim 256 \
    --actor-num-blocks 1
```

**Mở tủ (Cabinet) - SimbaV2:**
```bash
python fast_td3/train.py \
    --env-name h1hand-cabinet-v0 \
    --exp-name FastTD3_SimbaV2_Cabinet_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune \
    --agent fasttd3_simbav2 \
    --batch-size 8192 \
    --critic-learning-rate-end 3e-5 \
    --actor-learning-rate-end 3e-5 \
    --weight-decay 0.0 \
    --critic-hidden-dim 512 \
    --critic-num-blocks 2 \
    --actor-hidden-dim 256 \
    --actor-num-blocks 1
```

**Mở cửa ra vào (Door) - SimbaV2:**
```bash
python fast_td3/train.py \
    --env-name h1hand-door-v0 \
    --exp-name FastTD3_SimbaV2_Door_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune \
    --agent fasttd3_simbav2 \
    --batch-size 8192 \
    --critic-learning-rate-end 3e-5 \
    --actor-learning-rate-end 3e-5 \
    --weight-decay 0.0 \
    --critic-hidden-dim 512 \
    --critic-num-blocks 2 \
    --actor-hidden-dim 256 \
    --actor-num-blocks 1
```

**Mở cửa sổ (Window) - SimbaV2:**
```bash
python fast_td3/train.py \
    --env-name h1hand-window-v0 \
    --exp-name FastTD3_SimbaV2_Window_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune \
    --agent fasttd3_simbav2 \
    --batch-size 8192 \
    --critic-learning-rate-end 3e-5 \
    --actor-learning-rate-end 3e-5 \
    --weight-decay 0.0 \
    --critic-hidden-dim 512 \
    --critic-num-blocks 2 \
    --actor-hidden-dim 256 \
    --actor-num-blocks 1
```

**Sắp xếp giá sách (Bookshelf) - SimbaV2:**
```bash
python fast_td3/train.py \
    --env-name h1hand-bookshelf-v0 \
    --exp-name FastTD3_SimbaV2_Bookshelf_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune \
    --agent fasttd3_simbav2 \
    --batch-size 8192 \
    --critic-learning-rate-end 3e-5 \
    --actor-learning-rate-end 3e-5 \
    --weight-decay 0.0 \
    --critic-hidden-dim 512 \
    --critic-num-blocks 2 \
    --actor-hidden-dim 256 \
    --actor-num-blocks 1
```

**Tương tác nhà bếp (Kitchen) - SimbaV2:**
```bash
python fast_td3/train.py \
    --env-name h1hand-kitchen-v0 \
    --exp-name FastTD3_SimbaV2_Kitchen_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune \
    --agent fasttd3_simbav2 \
    --batch-size 8192 \
    --critic-learning-rate-end 3e-5 \
    --actor-learning-rate-end 3e-5 \
    --weight-decay 0.0 \
    --critic-hidden-dim 512 \
    --critic-num-blocks 2 \
    --actor-hidden-dim 256 \
    --actor-num-blocks 1
```

**Dùng thìa (Spoon) - SimbaV2:**
```bash
python fast_td3/train.py \
    --env-name h1hand-spoon-v0 \
    --exp-name FastTD3_SimbaV2_Spoon_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune \
    --agent fasttd3_simbav2 \
    --batch-size 8192 \
    --critic-learning-rate-end 3e-5 \
    --actor-learning-rate-end 3e-5 \
    --weight-decay 0.0 \
    --critic-hidden-dim 512 \
    --critic-num-blocks 2 \
    --actor-hidden-dim 256 \
    --actor-num-blocks 1
```

**Bê thùng hàng (Package) - SimbaV2:**
```bash
python fast_td3/train.py \
    --env-name h1hand-package-v0 \
    --exp-name FastTD3_SimbaV2_Package_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune \
    --agent fasttd3_simbav2 \
    --batch-size 8192 \
    --critic-learning-rate-end 3e-5 \
    --actor-learning-rate-end 3e-5 \
    --weight-decay 0.0 \
    --critic-hidden-dim 512 \
    --critic-num-blocks 2 \
    --actor-hidden-dim 256 \
    --actor-num-blocks 1
```

**Chất hàng lên xe tải (Truck) - SimbaV2:**
```bash
python fast_td3/train.py \
    --env-name h1hand-truck-v0 \
    --exp-name FastTD3_SimbaV2_Truck_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune \
    --agent fasttd3_simbav2 \
    --batch-size 8192 \
    --critic-learning-rate-end 3e-5 \
    --actor-learning-rate-end 3e-5 \
    --weight-decay 0.0 \
    --critic-hidden-dim 512 \
    --critic-num-blocks 2 \
    --actor-hidden-dim 256 \
    --actor-num-blocks 1
```

**Dọn dẹp căn phòng (Room) - SimbaV2:**
```bash
python fast_td3/train.py \
    --env-name h1hand-room-v0 \
    --exp-name FastTD3_SimbaV2_Room_Test \
    --seed 1 \
    --project "FastTD3_HumanoidBench_Experiments" \
    --render-interval 5000 \
    --total-timesteps 100001 \
    --compile-mode max-autotune \
    --agent fasttd3_simbav2 \
    --batch-size 8192 \
    --critic-learning-rate-end 3e-5 \
    --actor-learning-rate-end 3e-5 \
    --weight-decay 0.0 \
    --critic-hidden-dim 512 \
    --critic-num-blocks 2 \
    --actor-hidden-dim 256 \
    --actor-num-blocks 1
```

---

### 💡 Ghi chú khi quan sát Log:

- **Cảnh báo Gym/NumPy 2.0:** Bạn có thể bỏ qua các cảnh báo màu đỏ liên quan đến "Gym has been unmaintained". Quá trình khởi tạo 128 môi trường song song sẽ mất khoảng 1-2 phút.

- **Autotune Triton:** Các dòng dạng `AUTOTUNE mm(32768x216, 216x1024)` xuất hiện ở khoảng 100 step đầu tiên là tính năng JIT compiler đang tự động tối ưu hóa mạng nơ-ron trên chip A100. Khi quá trình này xong, FPS huấn luyện sẽ tăng vọt.

- **Biểu đồ:** Click vào đường link W&B hiện ra trong Terminal để theo dõi điểm số theo thời gian thực.

---

## Bước 5: Thực nghiệm Tinh chỉnh Reward (A/B Testing)

Theo bài báo gốc, hàm reward mặc định (dùng cho PPO) thường làm FastTD3 có dáng đi vung tay giật cục. Để có dáng đi mượt mà, cần cấu hình hệ số phạt (penalty) mạnh hơn.

### Cách thực hiện:

1. Trong **File Browser** của Jupyter Lab, tìm đến file cấu hình môi trường của humanoid-bench nằm trong đường dẫn đã cài đặt (thường ở `/usr/local/lib/python3.10/site-packages/humanoid_bench/` hoặc thư mục source humanoid-bench tương ứng).

2. Tăng các hệ số phạt `action_penalty`, `velocity_penalty`, v.v. (Ví dụ: từ `-0.01` lên `-0.1`).

3. Chạy lại lệnh huấn luyện ở Bước 4 nhưng đổi tên `--exp-name` thành `FastTD3_Tuned_Reward` để vẽ thành một đường biểu đồ so sánh mới trên W&B.

---

---

## Tóm tắt Quy trình

| Bước | Mô tả | Vị trí |
|------|-------|--------|
| 1 | Tạo `modal_jupyter.py` | Máy tính cá nhân |
| 2 | Chạy `modal run modal_jupyter.py` | Máy tính cá nhân (Terminal) |
| 3 | Fix EGL + Đăng nhập W&B | Jupyter Lab Terminal |
| 4 | Chạy huấn luyện | Jupyter Lab Terminal |
| 5 | Tinh chỉnh reward & so sánh | Jupyter Lab Terminal |

---

