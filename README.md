# DoBA (Document-Based AI) Project

DoBA is an AI-powered document analysis and processing system that leverages GPU acceleration for enhanced performance. This repository contains all the necessary scripts to set up and run the DoBA project on various hardware configurations.

## Features

- GPU-accelerated AI processing (supports both NVIDIA and AMD GPUs)
- Document analysis and text extraction
- OCR (Optical Character Recognition) capabilities
- Web search integration
- Extensible architecture for custom plugins

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

## Contributing

Contributions to the DoBA project are welcome! Please feel free to submit pull requests or open issues for any bugs or feature requests.

## License

This project is licensed under the terms included in the LICENSE file.
