# DoBA (TKinter Based) AI Project

DoBA is an autonomously configured AI ran in TKinter, that allows you to build a conversation and store it locally on your system. DoBA was designed primary by AI (hint at the name, Designed only By AI). This AI can be ran with any model, and can be ran through any API service (not just LM studio). This AI aims for TRUE autonomy, relying on tokenized keyword extraction via RAG. 

## Features

- GPU-accelerated AI processing (supports both NVIDIA and AMD GPUs)
- Document analysis and text extraction (through OCR and Search functions; MAY BE A BIT BUGGY FYI)
- OCR (Optical Character Recognition) capabilities
- Web search integration
- Extensible architecture for custom plugins
- Tokenized semantic keyword extraction (for highly accurate, and efficient memory)
- Emotional context and memory
- Autonomous self learning, self-deduplication of memory.
- Customizable database
- Infinite conversational storage
- Locally ran, no internet required.


## System Requirements

- Linux-based operating system (Ubuntu 20.04 or later recommended)
- Python 3.8 or later
- For GPU acceleration:
  - NVIDIA GPU with CUDA support, OR
  - AMD GPU with ROCm support
  - CPU-only mode is also supported for systems without compatible GPUs

## Installation

### One-Step Setup (Recommended)

1. Make the setup script executable:
   ```
   chmod +x setup.sh
   ```

2. Run the setup script:
   ```
   sudo ./setup.sh
   ```
   This will:
   - Detect your OS and install basic dependencies
   - Auto-detect your GPU type and suggest the appropriate option
   - Allow you to choose between AMD ROCm, NVIDIA CUDA, or CPU-only mode
   - Install the selected GPU platform and required dependencies
   - Install Python packages with the appropriate GPU support
   - Create launcher scripts optimized for your selected platform

### Alternative Setup (Manual)

If you prefer a more manual approach, you can use the individual scripts:

1. Make the copy script executable:
   ```
   chmod +x copy_doba_files.sh
   ```

2. Run the copy script:
   ```
   ./copy_doba_files.sh
   ```
   This will:
   - Copy all DoBA project files to your current directory
   - Create a Python virtual environment (.venv)
   - Detect your GPU type and create appropriate launcher scripts

3. Install dependencies:
   ```
   sudo ./install_dependencies.sh
   ```
   This will:
   - Prompt you to choose between AMD ROCm, NVIDIA CUDA, or CPU-only mode
   - Install the selected GPU platform and required dependencies
   - Install Python packages with the appropriate GPU support
   - Create launcher scripts optimized for your selected platform

4. Run the DoBA application:
   ```
   ./launch_doba.sh
   ```
   or
   ```
   ./launch_gpu.sh
   ```
   Both scripts are identical and will run the application with the appropriate GPU configuration.

### GPU Support Details

#### NVIDIA GPUs
If you have an NVIDIA GPU, the installation script will:
- Install CUDA drivers and toolkit
- Configure PyTorch with CUDA support
- Create launcher scripts with NVIDIA-specific optimizations

#### AMD GPUs
If you have an AMD GPU, the installation script will:
- Install ROCm drivers and libraries
- Configure PyTorch with ROCm support
- Create launcher scripts with AMD-specific optimizations

#### CPU-only Mode
If you don't have a compatible GPU or prefer not to use GPU acceleration:
- Select the "No GPU support" option during installation
- The system will use CPU-only mode for all operations

## LM Studio Configuration

DoBA uses LM Studio for model selection and inference. Follow these steps to set up LM Studio:

1. Download and install LM Studio from [https://lmstudio.ai/](https://lmstudio.ai/)

2. Launch LM Studio and ensure it's running on port 1234 (the default port)

3. Download and set up at least one model in LM Studio

4. Start the local server in LM Studio by clicking on the "Local Server" tab and then "Start Server"

5. By default, DoBA will connect to LM Studio at `http://localhost:1234`. If your LM Studio is running on a different machine or port, you can configure the connection by:
   - Editing the launcher scripts (launch_doba.sh or launch_gpu.sh)
   - Uncommenting and modifying the LM Studio API endpoint lines:
     ```bash
     # export LMSTUDIO_API="http://your-lmstudio-server:1234/v1/chat/completions"
     # export LMSTUDIO_MODELS_API="http://your-lmstudio-server:1234/v1/models"
     ```

6. When you launch DoBA, it should now be able to connect to LM Studio and display the available models in the model selection dropdown

## Troubleshooting

If you encounter any issues:

1. Make sure you have Python 3 installed:
   ```
   python3 --version
   ```

2. Ensure you have the necessary permissions to create files and directories in your current location.

3. For GPU-related issues:
   - NVIDIA: Run `nvidia-smi` to verify that your GPU is detected
   - AMD: Run `rocminfo` to verify that your GPU is detected

4. If you encounter conflicts between CUDA and ROCm, you may need to uninstall one before installing the other. The installation script will guide you through this process.

5. If the copy script fails, you can manually copy the files from the source directory to your current project directory.

6. If you're having issues with model selection:
   - Make sure LM Studio is running and the server is started
   - Check that you can access the LM Studio API at http://localhost:1234/v1/models in your browser
   - If using a remote LM Studio server, ensure the server is accessible from your machine
   - If the model dropdown is blank, try clicking the "Refresh Models" button
   - If models still don't appear, check the Debug tab for error messages
   - The application will use placeholder models if it can't connect to LM Studio

## Contributing

Contributions to the DoBA project are welcome! Please feel free to submit pull requests or open issues for any bugs or feature requests.

## License

This project is licensed under the terms included in the LICENSE file.
