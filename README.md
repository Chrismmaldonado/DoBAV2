arkdown

# DoBA (TKinter Based) AI Project

DoBA is an AI program that runs in TKinter and allows you to create conversations and save them locally on your system. It was mainly designed by AI (the name hints at that). You can run this AI with any model and through any API service, not just LM Studio. The goal of this AI is to achieve true independence by using tokenized keyword extraction through RAG.

---

## Features

* **GPU-accelerated AI processing:** Compatible with NVIDIA and AMD GPUs.
* **Document analysis and text extraction:** Using OCR and search functions.
* **OCR (Optical Character Recognition):** Capabilities for reading text from images.
* **Web search integration:** Access to external information.
* **Custom plugins:** Extensible framework for adding new functionality.
* **Tokenized semantic keyword extraction:** Used for effective memory retrieval.
* **Emotional context:** Tracks and utilizes emotional history.
* **Self-learning:** Includes self-deduplication of memory entries.
* **Customizable database:** Flexible storage options.
* **Unlimited conversation storage:** Runs locally with no internet needed for operation.

---

## System Requirements

* **Operating System:** Linux-based (Ubuntu 20.04 or later recommended).
* **Python:** Version 3.8 or later.
* **GPU Acceleration (Optional):**
    * NVIDIA GPU with CUDA support.
    * AMD GPU with ROCm support.
    * CPU-only mode is supported for systems without compatible GPUs.

---

## Installation

### One-Step Setup (Recommended)

1. **Make the setup script executable:**
   ```sh
   chmod +x setup.sh
Run the setup script:

Bash

sudo ./setup.sh
The script will:

Detect your OS and install basic dependencies.

Auto-detect your GPU type and suggest the right option.

Allow you to choose between AMD ROCm, NVIDIA CUDA, or CPU-only mode.

Install Python packages with the correct GPU support.

Create optimized launcher scripts.

Alternative Setup (Manual)
Make the copy script executable:

Bash

chmod +x copy_doba_files.sh
Run the copy script:

Bash

./copy_doba_files.sh
Copies files, creates a virtual environment (.venv), and detects GPU.

Install dependencies:

Bash

sudo ./install_dependencies.sh
Follow the prompts for your specific hardware (NVIDIA/AMD/CPU).

Run the DoBA application:

Bash

./launch_doba.sh
or

Bash

./launch_gpu.sh
GPU Support Details
NVIDIA GPUs
The installation script will install CUDA drivers and toolkit, configure PyTorch with CUDA support, and apply NVIDIA-specific optimizations.

AMD GPUs
The installation script will install ROCm drivers and libraries, configure PyTorch with ROCm support, and apply AMD-specific optimizations.

CPU-only Mode
If you lack a compatible GPU, choose the "No GPU support" option. The system will use CPU-only mode for all tasks.

LM Studio Configuration
DoBA uses LM Studio for selecting and running models.

Download and install LM Studio from https://lmstudio.ai/.

Ensure LM Studio is running on the default port 1234.

Download at least one model within LM Studio.

Go to the "Local Server" tab and click "Start Server".

By default, DoBA connects to http://localhost:1234. To change this, edit the launcher scripts and modify the following variables:

Bash

# export LMSTUDIO_API="http://your-lmstudio-server:1234/v1/chat/completions"
# export LMSTUDIO_MODELS_API="http://your-lmstudio-server:1234/v1/models"
Troubleshooting
Check Python version:

Bash

python3 --version
Check Permissions: Ensure you have write access to the project directory.

GPU Detection:

NVIDIA: Run nvidia-smi.

AMD: Run rocminfo.

Conflicts: If you face conflicts between CUDA and ROCm, uninstall the previous drivers before switching.

Model Selection Issues:

Ensure the LM Studio server is started.

Verify access to http://localhost:1234/v1/models in your browser.

If the model dropdown is empty, click "Refresh Models" or check the Debug tab for errors.

Contributing
Contributions are welcome. Feel free to submit pull requests or report bugs and feature requests via the issue tracker.
