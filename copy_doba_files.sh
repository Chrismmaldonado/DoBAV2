#!/bin/bash

# Script to copy DoBA project files to the current project and set up the environment

# Get the current directory (destination)
DEST_DIR=$(pwd)
echo "Destination directory: $DEST_DIR"

# Source directory (DoBA project)
SOURCE_DIR="/home/chris/DoBAv2"
echo "Source directory: $SOURCE_DIR"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
  echo "Error: Source directory $SOURCE_DIR does not exist."
  exit 1
fi

# Create a virtual environment in the destination directory if it doesn't exist
if [ ! -d ".venv" ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv .venv
  echo "✅ Virtual environment created."
fi

# Copy all project files from the source directory to the destination directory
echo "Copying project files from $SOURCE_DIR to $DEST_DIR..."
cp -r $SOURCE_DIR/* $DEST_DIR/

# Make the install_dependencies.sh script executable
chmod +x $DEST_DIR/install_dependencies.sh

# Create launcher scripts that use the local virtual environment
echo "Creating launcher scripts..."

# Detect GPU type
echo "Detecting GPU type..."
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU detected. Creating CUDA launcher scripts."
  GPU_TYPE="nvidia"
elif command -v rocminfo >/dev/null 2>&1; then
  echo "AMD GPU detected. Creating ROCm launcher scripts."
  GPU_TYPE="amd"
else
  echo "No GPU detected or drivers not installed. Creating CPU-only launcher scripts."
  GPU_TYPE="cpu"
fi

if [ "$GPU_TYPE" = "amd" ]; then
  # Create AMD ROCm launcher scripts
  cat > $DEST_DIR/launch_doba.sh << 'EOF'
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

  cat > $DEST_DIR/launch_gpu.sh << 'EOF'
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

elif [ "$GPU_TYPE" = "nvidia" ]; then
  # Create NVIDIA CUDA launcher scripts
  cat > $DEST_DIR/launch_doba.sh << 'EOF'
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

  cat > $DEST_DIR/launch_gpu.sh << 'EOF'
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
  cat > $DEST_DIR/launch_doba.sh << 'EOF'
#!/bin/bash
# CPU-only configuration
# No GPU-specific environment variables needed

# Activate virtual environment
source .venv/bin/activate

# Run DoBA with extensions
python DobAEI.py
EOF

  cat > $DEST_DIR/launch_gpu.sh << 'EOF'
#!/bin/bash
# CPU-only configuration
# No GPU-specific environment variables needed

# Activate virtual environment
source .venv/bin/activate

# Run DoBA with extensions
python DobAEI.py
EOF
fi

chmod +x $DEST_DIR/launch_doba.sh
chmod +x $DEST_DIR/launch_gpu.sh

echo ""
echo "===== DoBA Files Copied Successfully ====="
echo "To install dependencies, run: sudo ./install_dependencies.sh"
echo "To run DoBA with extensions, use: ./launch_doba.sh or ./launch_gpu.sh"
echo ""
