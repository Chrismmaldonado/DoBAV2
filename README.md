# DoBA (TKinter Based) AI Project

DoBA is an AI program that runs in **TKinter**. It allows you to create conversations and save them locally on your system. AI mainly designed this program. You can use **any model and API service**, not just LM Studio. This AI aims to achieve **true independence through tokenized keyword extraction with RAG**.

---

# Features

* **GPU-accelerated AI processing** – Works with NVIDIA and AMD GPUs.
* **Document analysis and text extraction** – Uses OCR and search functions; this feature may be experimental.
* **OCR (Optical Character Recognition)** – Can read text from visual data.
* **Web search integration** – Can pull external information.
* **Custom plugins** – Flexible framework for adding new features.
* **Tokenized semantic keyword extraction** – Helps with effective memory retrieval.
* **Emotional context and memory** – Tracks and uses emotional history for responses.
* **Self-learning and self-deduplication** – Automatically manages and cleans memory entries.
* **Customizable database** – Flexible storage options for user data.
* **Unlimited conversation storage** – Works locally without needing an internet connection.

---

# System Requirements

* **Operating System:** Linux-based (Ubuntu 20.04 or later recommended)
* **Python:** Version 3.8 or later

### GPU Acceleration

* **NVIDIA GPU** with CUDA support
* **AMD GPU** with ROCm support

### CPU-only Mode

* Supported for systems without compatible GPUs.

---

# Installation

## One-Step Setup (Recommended)

### 1. Make the setup script executable

```bash
chmod +x setup.sh
```

### 2. Run the setup script

```bash
sudo ./setup.sh
```

### This script will

* Detect your OS and install basic dependencies.
* Auto-detect your GPU type and suggest the right option.
* Let you choose between **AMD ROCm**, **NVIDIA CUDA**, or **CPU-only mode**.
* Install the selected GPU platform and necessary dependencies.
* Install Python packages with the correct GPU support.
* Create launcher scripts optimized for your selected platform.

---

# Alternative Setup (Manual)

### 1. Make the copy script executable

```bash
chmod +x copy_doba_files.sh
```

### 2. Run the copy script

```bash
./copy_doba_files.sh
```

This will:

* Copy project files
* Create a Python virtual environment (`.venv`)
* Detect GPU type

---

### 3. Install dependencies

```bash
sudo ./install_dependencies.sh
```

This script will prompt you to choose:

* ROCm
* CUDA
* CPU-only

Then it installs the relevant packages.

---

### 4. Run the DoBA application

```bash
./launch_doba.sh
```

or

```bash
./launch_gpu.sh
```

---

# GPU Support Details

## NVIDIA GPUs

The installation script will:

* Install **CUDA drivers and toolkit**
* Configure **PyTorch with CUDA support**
* Create launcher scripts optimized for **NVIDIA GPUs**

---

## AMD GPUs

The installation script will:

* Install **ROCm drivers and libraries**
* Configure **PyTorch with ROCm support**
* Create launcher scripts optimized for **AMD GPUs**

---

## CPU-only Mode

If you do not have a compatible GPU:

* Choose **"No GPU support"** during installation.
* The system will run in **CPU-only mode** for all tasks.

---

# LM Studio Configuration

DoBA uses **LM Studio** for selecting and running models.

### Setup Steps

1. Download and install LM Studio from
   https://lmstudio.ai/

2. Ensure LM Studio is running on **port 1234** (default).

3. Download and set up **at least one model**.

4. Start the **local server** in the **Local Server tab**.

---

### Default Connection

By default, DoBA connects to:

```
http://localhost:1234
```

To use a **different machine or port**, modify the launcher scripts:

```bash
export LMSTUDIO_API="http://your-lmstudio-server:1234/v1/chat/completions"
export LMSTUDIO_MODELS_API="http://your-lmstudio-server:1234/v1/models"
```

Available models will appear in the **selection dropdown** when the program launches.

---

# Troubleshooting

### Verify Python

Make sure Python 3 is installed:

```bash
python3 --version
```

---

### Permissions

Ensure you have the correct permissions to:

* Create files
* Create directories
* Run scripts

---

### GPU Detection

**NVIDIA**

```bash
nvidia-smi
```

**AMD**

```bash
rocminfo
```

---

### Driver Conflicts

If switching between **CUDA and ROCm**, uninstall conflicting GPU platforms first.

---

### Manual File Copy

If the copy script fails:

* Manually move files from the **source directory** to the **project directory**.

---

### Model Selection Problems

If models do not appear:

* Verify the **LM Studio server is running**.
* Check browser access:

```
http://localhost:1234/v1/models
```

* Use the **Refresh Models** button if the dropdown is empty.
* Check the **Debug tab** for specific error logs.

---

# Contributing

You can contribute to the **DoBA project**.

Ways to help:

* Submit **pull requests**
* Report **bugs**
* Suggest **feature requests**

All improvements are welcome.

