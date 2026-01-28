#!/usr/bin/env python3
"""Setup script for HOI detection project."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("Setting up Human-Object Interaction Detection Project")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("Error: Python 3.10+ is required")
        sys.exit(1)
    
    print(f"Python version: {sys.version}")
    
    # Install package in development mode
    if not run_command("pip install -e .", "Installing package in development mode"):
        sys.exit(1)
    
    # Install development dependencies
    if not run_command("pip install -e .[dev]", "Installing development dependencies"):
        print("Warning: Some development dependencies failed to install")
    
    # Install Detectron2 (optional)
    print("\nInstalling Detectron2 (optional)...")
    detectron2_cmd = "pip install 'git+https://github.com/facebookresearch/detectron2.git'"
    if run_command(detectron2_cmd, "Installing Detectron2"):
        print("✓ Detectron2 installed successfully")
    else:
        print("⚠ Detectron2 installation failed - some features may not work")
    
    # Create necessary directories
    directories = ["data", "checkpoints", "results", "assets"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Run tests
    print("\nRunning tests...")
    if run_command("python -m pytest tests/ -v", "Running unit tests"):
        print("✓ All tests passed")
    else:
        print("⚠ Some tests failed - check the output above")
    
    # Format code
    print("\nFormatting code...")
    if run_command("black src/ tests/", "Formatting code with black"):
        print("✓ Code formatted successfully")
    
    # Lint code
    if run_command("ruff check src/ tests/", "Linting code with ruff"):
        print("✓ Code linting passed")
    else:
        print("⚠ Code linting found issues - check the output above")
    
    print("\n" + "=" * 60)
    print("Setup completed!")
    print("\nNext steps:")
    print("1. Download HICO-DET or V-COCO dataset to the 'data/' directory")
    print("2. Run training: python scripts/train.py --data_root data/hico_20160224_det")
    print("3. Run evaluation: python scripts/eval.py --checkpoint checkpoints/best_model.pth --data_root data/hico_20160224_det")
    print("4. Launch demo: streamlit run demo/app.py")
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
