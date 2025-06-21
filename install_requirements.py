#!/usr/bin/env python3

import subprocess
import sys

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {command}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ {command}")
        print(f"Error: {e.stderr}")
        return None

def check_cuda_availability():
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            return True
        else:
            print("❌ No NVIDIA GPU detected")
            return False
    except:
        print("❌ nvidia-smi not found - No NVIDIA GPU support")
        return False

def install_pytorch_cuda():
    print("\n🔧 Installing PyTorch with CUDA support...")
    
    print("Uninstalling existing PyTorch...")
    run_command("pip uninstall torch torchvision torchaudio -y")
    
    cuda_command = "pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118"
    result = run_command(cuda_command)
    
    if result:
        print("✅ PyTorch with CUDA installed successfully")
        return True
    else:
        print("❌ Failed to install PyTorch with CUDA")
        return False

def install_pytorch_cpu():
    print("\n🔧 Installing PyTorch CPU version...")
    
    run_command("pip uninstall torch torchvision torchaudio -y")
    
    cpu_command = "pip install torch==2.7.1+cpu torchvision==0.22.1+cpu torchaudio==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu"
    result = run_command(cpu_command)
    
    if result:
        print("✅ PyTorch CPU version installed successfully")
        return True
    else:
        print("❌ Failed to install PyTorch CPU version")
        return False

def install_other_requirements():
    print("\n📦 Installing other requirements...")
    
    requirements = [
        "fastapi==0.104.1",
        "uvicorn==0.24.0", 
        "requests==2.31.0",
        "azure-storage-blob==12.17.0",
        "python-dotenv==1.0.0",
        "pydantic==2.5.2",
        "pillow==10.1.0",
        "numpy==1.24.3",
        "urllib3==2.1.0"
    ]
    
    for req in requirements:
        run_command(f"pip install {req}")

def verify_installation():
    print("\n🔍 Verifying PyTorch installation...")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"✅ GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("ℹ️  Running on CPU mode")
            
        return True
    except Exception as e:
        print(f"❌ PyTorch verification failed: {e}")
        return False

def main():
    print("🚀 WriterID Backend Installation Script")
    print("=" * 50)
    
    has_cuda = check_cuda_availability()
    
    if has_cuda:
        success = install_pytorch_cuda()
        if not success:
            print("⚠️  CUDA installation failed, falling back to CPU version")
            success = install_pytorch_cpu()
    else:
        print("ℹ️  No CUDA support detected, installing CPU version")
        success = install_pytorch_cpu()
    
    if not success:
        print("❌ Failed to install PyTorch")
        sys.exit(1)
    
    install_other_requirements()
    
    if verify_installation():
        print("\n🎉 Installation completed successfully!")
        print("You can now run: python app.py")
    else:
        print("\n❌ Installation verification failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 