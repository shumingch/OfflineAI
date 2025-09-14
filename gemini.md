# Gemini CLI Project Documentation

This document outlines the development process and key information for the Offline AI Image Generator project, as guided by the Gemini CLI.

## Project Goal
To create a free, offline AI image generator using Stable Diffusion, leveraging local GPU resources, with a user-friendly web interface.

## Key Features
- **Offline Generation:** All AI model inference runs locally on your NVIDIA GPU.
- **Two-Stage Upscaling:** Generates a high-quality base image (768x768) and then upscales it to a higher resolution (1536x1536) for improved detail and quality.
- **Web User Interface:** A Gradio-based web interface for easy interaction without command-line arguments.
- **Dependency Isolation:** Utilizes separate Python virtual environments to manage conflicting library dependencies.

## Project Structure
- `.venv-pt/`: Python virtual environment for the core image generation logic (PyTorch, Diffusers, Transformers).
- `.venv-ui/`: Python virtual environment for the Gradio web interface and its dependencies.
- `generate_image.py`: The core Python script for image generation and upscaling. It's designed to be called as a subprocess.
- `app.py`: The Gradio web application script that provides the user interface and orchestrates calls to `generate_image.py`.
- `requirements.txt`: Lists the exact Python package versions for the `.venv-pt` environment.
- `README.md`: Project overview and setup instructions.

## Setup and Running Instructions

### Prerequisites
- Python 3.11 installed on your system.
- An NVIDIA GPU with compatible drivers.
- `git` and `gh` (GitHub CLI) installed (for initial setup).

### 1. Clone the Repository
```bash
git clone https://github.com/shumingch/OfflineAI.git
cd OfflineAI
```

### 2. Set up Python Environments

#### Core Image Generation Environment (`.venv-pt`)
This environment contains the heavy AI libraries.
```bash
py -3.11 -m venv .venv-pt

# Activate the environment (optional, but good for direct interaction)
# On Windows: .venv-pt\Scripts\activate

# Install dependencies (specific versions for compatibility)
.venv-pt\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
.venv-pt\Scripts\pip install -r requirements.txt
```
*(Note: `requirements.txt` will be updated by Gemini CLI to reflect the exact pinned versions.)*

#### Web UI Environment (`.venv-ui`)
This environment contains the Gradio web framework.
```bash
py -3.11 -m venv .venv-ui

# Activate the environment (optional)
# On Windows: .venv-ui\Scripts\activate

# Install Gradio
.venv-ui\Scripts\pip install gradio
```

### 3. Launch the Web Application
Once both environments are set up, you can launch the Gradio web interface. This script will automatically manage calling the image generation script in its separate environment.

```bash
.venv-ui\Scripts\python.exe app.py
```
Open the URL provided in your terminal (usually `http://127.0.0.1:7860`) in your web browser.

## Development Notes
- **Dependency Management:** Due to conflicting dependencies between `diffusers` (requiring older `huggingface-hub` versions) and `gradio` (requiring newer `huggingface-hub` and `pydantic` v2), a multi-environment strategy was adopted.
- **Subprocess Communication:** The `app.py` script communicates with `generate_image.py` via command-line arguments and temporary file storage using Python's `subprocess` module.
