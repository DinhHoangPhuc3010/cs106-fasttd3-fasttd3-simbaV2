import modal
import subprocess

app = modal.App("fasttd3-a100-workspace")

# Cấu hình Image theo chuẩn README
fasttd3_image = (
    modal.Image.debian_slim(python_version="3.10")
    # Cài đặt các system packages cần thiết
    .apt_install(
        "libglfw3", "libgl1-mesa-glx", "libosmesa6", 
        "git-lfs", "cmake", "xvfb", "ffmpeg", "git"
    )
    # Cài đặt các công cụ Python cơ bản
    .pip_install(
        "jupyterlab", "stable-baselines3", "wandb", "tensorboard", "pyvirtualdisplay"
    )
    .run_commands(
        # 1. Cài PyTorch hỗ trợ CUDA 12.1 (tối ưu cho A100)
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        
        # 2. Cài HumanoidBench (baseline) theo đúng lệnh trong README
        "pip install --editable git+https://github.com/carlosferrazza/humanoid-bench.git#egg=humanoid-bench",
        
        # # 2.1 Cài HumanoidBench (cho FastTD3) theo đúng lệnh trong README
        # "pip install --editable git+https://github.com/DinhHoangPhuc3010/humanoid-bench-fasttd3.git#egg=humanoid-bench"

        # 3. Clone repository FastTD3
        "git clone https://github.com/younggyoseo/FastTD3.git /workspace/FastTD3",
        
        # 4. Cài đặt requirements của FastTD3
        "cd /workspace/FastTD3 && pip install -r requirements/requirements.txt"
    )
)

# Khởi chạy ứng dụng trên 1 GPU A100 và 16 CPU cores
@app.function(image=fasttd3_image, gpu="A100", cpu=16, timeout=86400)
def run_jupyter():
    import modal # Gọi thêm modal bên trong hàm
    print("Đang khởi tạo đường hầm kết nối...")
    
    # Mở đường hầm ở port 8888 ra internet để máy bạn có thể truy cập
    with modal.forward(8888) as tunnel:
        print(f"\n🚀 TRUY CẬP JUPYTER LAB TẠI LINK SAU: {tunnel.url}\n")
        
        # Khởi chạy Jupyter Lab
        subprocess.run([
            "jupyter", "lab", 
            "--ip=0.0.0.0", 
            "--port=8888", 
            "--allow-root", 
            "--no-browser", 
            "--notebook-dir=/workspace/FastTD3",
            "--ServerApp.token=''",    # Bỏ yêu cầu nhập mật khẩu để vào thẳng
            "--ServerApp.password=''"
        ])