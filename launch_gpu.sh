#!/bin/bash
# AMD ROCm environment variables
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export AMD_SERIALIZE_KERNEL=3
export HIP_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100

# LM Studio API endpoints (uncomment and modify if needed)
# export LMSTUDIO_API="http://your-lmstudio-server:1234/v1/chat/completions"
# export LMSTUDIO_MODELS_API="http://your-lmstudio-server:1234/v1/models"

# Activate virtual environment
source .venv/bin/activate

# Run DoBA with extensions
python DobAEI.py
