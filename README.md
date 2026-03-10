# DoBA (TKinter Based) AI Project

DoBA is an AI program that runs in TKinter and allows you to create conversations and save them locally on your system. It was mainly designed by AI (the name hints at that). You can run this AI with any model and through any API service, not just LM Studio. The goal of this AI is to achieve true independence by using tokenized keyword extraction through RAG.

---

## Features

- **GPU-accelerated AI processing:** Compatible with NVIDIA and AMD GPUs.
- **Document analysis and text extraction:** Uses OCR and search functions; note that this feature may be experimental.
- **OCR (Optical Character Recognition):** Capabilities for reading text from visual data.
- **Web search integration:** Capability to pull external information.
- **Custom plugins:** Extensible framework for adding new functionality.
- **Tokenized semantic keyword extraction:** Enables effective memory retrieval.
- **Emotional context and memory:** Tracks and utilizes emotional history for responses.
- **Self-learning and self-deduplication:** Automatically manages and cleans memory entries.
- **Customizable database:** Flexible storage options for user data.
- **Unlimited conversation storage:** Runs locally with no internet needed for operation.

---

## System Requirements

- **Operating System:** Linux-based (Ubuntu 20.04 or later recommended).
- **Python:** Version 3.8 or later.
- **For GPU acceleration:**
  - NVIDIA GPU with CUDA support.
  - AMD GPU with ROCm support.
- **CPU-only mode:** Supported for systems without compatible GPUs.

---

## Installation

### One-Step Setup (Recommended)

1. **Make the setup script executable:**
   ```sh
   chmod +x setup.sh
   ```

2. **Run the setup script:**
   ```sh
   sudo ./setup.sh
   ```

This script will:

- Detect your OS and install basic dependencies.
- Auto-detect your GPU type and suggest the right option.
- Let you choose between AMD ROCm, NVIDIA CUDA, or CPU-only mode.
- Install the selected GPU platform and necessary dependencies.
- Install Python packages with the correct GPU support.
- Create launcher scripts optimized for your selected platform.

### Alternative Setup (Manual)

1. **Make the copy script executable:**
   ```sh
   chmod +x copy_doba_files.sh
   ```

2. **Run the copy script:**
   ```sh
   ./copy_doba_files.sh
   ```
   Copies project files, creates a Python virtual environment (`.venv`), and detects GPU type.

3. **Install dependencies:**
   ```sh
   sudo ./install_dependencies.sh
   ```
   Prompts for hardware choice (ROCm/CUDA/CPU) and installs relevant packages.

4. **Run the DoBA application:**
   ```sh
   ./launch_doba.sh
   ```
   or
   ```sh
   ./launch_gpu.sh
   ```

---

## GPU Support Details

### NVIDIA GPUs

The installation script installs CUDA drivers and toolkit, configures PyTorch with CUDA support, and creates launcher scripts with NVIDIA-specific optimizations.

### AMD GPUs

The installation script installs ROCm drivers and libraries, configures PyTorch with ROCm support, and creates launcher scripts with AMD-specific optimizations.

### CPU-only Mode

If you lack a compatible GPU, choose the "No GPU support" option during installation. The system will use CPU-only mode for all tasks.

---

## LM Studio Configuration

DoBA uses LM Studio for selecting and running models.

1. Download and install LM Studio from [https://lmstudio.ai/](https://lmstudio.ai/).
2. Ensure LM Studio is running on port `1234` (default).
3. Download and set up at least one model.
4. Start the local server in the **Local Server** tab.

By default, DoBA connects to `http://localhost:1234`. To use a different machine or port, modify the launcher scripts:

```sh
# export LMSTUDIO_API="http://your-lmstudio-server:1234/v1/chat/completions"
# export LMSTUDIO_MODELS_API="http://your-lmstudio-server:1234/v1/models"
```

Available models will appear in the selection dropdown upon launch.

---

## Troubleshooting

- **Verify Python:** Ensure Python 3 is installed via `python3 --version`.
- **Permissions:** Ensure you have the right permissions to create files and directories.
- **GPU Detection:**
  - NVIDIA: Run `nvidia-smi`.
  - AMD: Run `rocminfo`.
- **Driver Conflicts:** Uninstall conflicting GPU platforms before switching between CUDA and ROCm.
- **Manual Copy:** If the copy script fails, manually move files from the source to the project directory.
- **Model Selection:**
  - Verify the server is started in LM Studio.
  - Check browser access to `http://localhost:1234/v1/models`.
  - Use the **Refresh Models** button if the dropdown is empty.
  - Check the **Debug** tab for specific error logs.

---

## Contributing

You can contribute to the DoBA project! Feel free to submit pull requests or report bugs and feature requests.
