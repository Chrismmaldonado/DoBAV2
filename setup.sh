#!/bin/bash
# DoBA Project Setup Script
# This script automates the setup process for the DoBA project

echo "===== DoBA Project Setup ====="
echo "This script will set up the DoBA project with all required dependencies."
echo "It requires sudo privileges for system package installation."
echo ""

# Check if script is run with sudo
if [ "$EUID" -ne 0 ]; then
  echo "Please run this script with sudo privileges."
  echo "Usage: sudo ./setup.sh"
  exit 1
fi

# Get the current directory
DEST_DIR=$(pwd)
echo "Installation directory: $DEST_DIR"

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Create a virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv .venv
  echo "✅ Virtual environment created."
fi

# Detect OS
if [ -f /etc/os-release ]; then
  . /etc/os-release
  OS=$NAME
  VER=$VERSION_ID
else
  echo "❌ Cannot detect OS. This script supports Ubuntu and similar distributions."
  exit 1
fi

echo "Detected OS: $OS $VER"

# Update package lists
echo "Updating package lists..."
apt-get update

# Install basic dependencies
echo "Installing basic dependencies..."
apt-get install -y \
  python3-pip \
  python3-dev \
  build-essential \
  git \
  curl \
  wget

# Install Tesseract OCR
echo "Installing Tesseract OCR..."
apt-get install -y \
  tesseract-ocr \
  libtesseract-dev \
  libleptonica-dev \
  tesseract-ocr-eng

# Function to install ROCm
install_rocm() {
  # Check if ROCm is already installed
  if command_exists rocminfo; then
    echo "✅ ROCm is already installed."
    return
  fi

  # Check if CUDA is installed and uninstall if necessary
  if command_exists nvidia-smi; then
    echo "⚠️ NVIDIA CUDA is currently installed. ROCm and CUDA cannot be used together."
    read -p "Do you want to uninstall CUDA to install ROCm? (y/n): " uninstall_cuda
    if [ "$uninstall_cuda" = "y" ]; then
      echo "Uninstalling CUDA..."
      apt-get remove --purge -y '*nvidia*' '*cuda*'
      apt-get autoremove -y
      echo "✅ CUDA uninstalled."
    else
      echo "❌ ROCm installation aborted. CUDA will remain installed."
      return
    fi
  fi

  echo "Installing ROCm for AMD GPUs..."

  # Add ROCm repository
  wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
  echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main" | tee /etc/apt/sources.list.d/rocm.list
  apt-get update

  # Install ROCm packages
  apt-get install -y \
    rocm-dev \
    rocm-libs \
    rocm-utils \
    rocm-cmake

  # Add user to video group for ROCm access
  usermod -a -G video $SUDO_USER
  usermod -a -G render $SUDO_USER

  # Set up environment variables
  echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/rocprofiler/bin:/opt/rocm/opencl/bin' | tee -a /etc/profile.d/rocm.sh
  echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib:/opt/rocm/lib64' | tee -a /etc/profile.d/rocm.sh
  chmod +x /etc/profile.d/rocm.sh

  echo "✅ ROCm installation completed."
}

# Function to install CUDA
install_cuda() {
  # Check if CUDA is already installed
  if command_exists nvidia-smi; then
    echo "✅ NVIDIA CUDA is already installed."
    return
  fi

  # Check if ROCm is installed and uninstall if necessary
  if command_exists rocminfo; then
    echo "⚠️ AMD ROCm is currently installed. ROCm and CUDA cannot be used together."
    read -p "Do you want to uninstall ROCm to install CUDA? (y/n): " uninstall_rocm
    if [ "$uninstall_rocm" = "y" ]; then
      echo "Uninstalling ROCm..."
      apt-get remove --purge -y '*rocm*' 'hip*'
      apt-get autoremove -y
      rm -f /etc/apt/sources.list.d/rocm.list
      echo "✅ ROCm uninstalled."
    else
      echo "❌ CUDA installation aborted. ROCm will remain installed."
      return
    fi
  fi

  echo "Installing NVIDIA CUDA..."

  # Add CUDA repository
  wget -q -O /etc/apt/preferences.d/cuda-repository-pin-600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
  apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
  add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
  apt-get update

  # Install CUDA packages
  apt-get install -y cuda-toolkit-11-8

  # Set up environment variables
  echo 'export PATH=$PATH:/usr/local/cuda/bin' | tee -a /etc/profile.d/cuda.sh
  echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64' | tee -a /etc/profile.d/cuda.sh
  chmod +x /etc/profile.d/cuda.sh

  echo "✅ CUDA installation completed."
}

# Auto-detect GPU type for default selection
DEFAULT_CHOICE="3"  # Default to CPU-only
if command_exists nvidia-smi; then
  DEFAULT_CHOICE="2"  # NVIDIA detected
  echo "NVIDIA GPU detected."
elif command_exists rocminfo; then
  DEFAULT_CHOICE="1"  # AMD detected
  echo "AMD GPU detected."
else
  echo "No GPU detected or drivers not installed."
fi

# Ask user which GPU platform to install
echo ""
echo "Please select which GPU platform to install:"
echo "1) AMD ROCm (for AMD GPUs)"
echo "2) NVIDIA CUDA (for NVIDIA GPUs)"
echo "3) No GPU support (CPU only)"
read -p "Enter your choice (1-3) [Default: $DEFAULT_CHOICE]: " gpu_choice

# Use default if no input provided
if [ -z "$gpu_choice" ]; then
  gpu_choice=$DEFAULT_CHOICE
fi

case $gpu_choice in
  1)
    install_rocm
    ;;
  2)
    install_cuda
    ;;
  3)
    echo "Skipping GPU platform installation."
    ;;
  *)
    echo "Invalid choice. Skipping GPU platform installation."
    ;;
esac

# Install Python packages
echo "Installing Python packages..."

# Activate virtual environment
source .venv/bin/activate

# Install PyTorch with appropriate GPU support
if [ "$gpu_choice" = "1" ]; then
  echo "Installing PyTorch with ROCm support..."
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
elif [ "$gpu_choice" = "2" ]; then
  echo "Installing PyTorch with CUDA support..."
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
  echo "Installing PyTorch with CPU support..."
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install OCR packages
echo "Installing OCR Python packages..."
pip install pytesseract Pillow

# Install web search packages
echo "Installing web search packages..."
pip install duckduckgo_search

# Install other required packages
echo "Installing other required packages..."
pip install requests beautifulsoup4 "numpy<2.0"

# Create launcher scripts
echo "Creating launcher scripts..."

if [ "$gpu_choice" = "1" ]; then
  # Create AMD ROCm launcher scripts
  cat > launch_doba.sh << 'EOF'
#!/bin/bash
# AMD ROCm environment variables
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export AMD_SERIALIZE_KERNEL=3
export HIP_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100

# Activate virtual environment
source .venv/bin/activate

# Run DoBA with extensions
python DobAEI.py
EOF

  cat > launch_gpu.sh << 'EOF'
#!/bin/bash
# AMD ROCm environment variables
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export AMD_SERIALIZE_KERNEL=3
export HIP_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100

# Activate virtual environment
source .venv/bin/activate

# Run DoBA with extensions
python DobAEI.py
EOF

elif [ "$gpu_choice" = "2" ]; then
  # Create NVIDIA CUDA launcher scripts
  cat > launch_doba.sh << 'EOF'
#!/bin/bash
# NVIDIA CUDA environment variables
export CUDA_VISIBLE_DEVICES=0
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_MAXSIZE=2147483648
export CUDA_CACHE_PATH=~/.cuda_cache

# Activate virtual environment
source .venv/bin/activate

# Run DoBA with extensions
python DobAEI.py
EOF

  cat > launch_gpu.sh << 'EOF'
#!/bin/bash
# NVIDIA CUDA environment variables
export CUDA_VISIBLE_DEVICES=0
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_MAXSIZE=2147483648
export CUDA_CACHE_PATH=~/.cuda_cache

# Activate virtual environment
source .venv/bin/activate

# Run DoBA with extensions
python DobAEI.py
EOF

else
  # Create CPU-only launcher scripts
  cat > launch_doba.sh << 'EOF'
#!/bin/bash
# CPU-only configuration
# No GPU-specific environment variables needed

# Activate virtual environment
source .venv/bin/activate

# Run DoBA with extensions
python DobAEI.py
EOF

  cat > launch_gpu.sh << 'EOF'
#!/bin/bash
# CPU-only configuration
# No GPU-specific environment variables needed

# Activate virtual environment
source .venv/bin/activate

# Run DoBA with extensions
python DobAEI.py
EOF
fi

chmod +x launch_doba.sh
chmod +x launch_gpu.sh

echo ""
echo "===== DoBA Setup Complete ====="
echo "To run DoBA, use: ./launch_doba.sh or ./launch_gpu.sh"
echo ""
echo "Note: You may need to log out and log back in for GPU driver changes to take effect."
echo "If you encounter any issues, please check the README.md file for troubleshooting tips."
