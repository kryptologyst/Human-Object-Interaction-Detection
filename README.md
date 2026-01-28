# Human-Object Interaction Detection

Advanced Computer Vision project for detecting and analyzing human-object interactions in images and videos.

## Overview

This project implements state-of-the-art Human-Object Interaction (HOI) detection models, focusing on identifying relationships between humans and objects in visual scenes. The system can detect not only the presence of humans and objects but also their specific interactions (e.g., "person holding cup", "person sitting on chair").

## Features

- **Modern HOI Detection Models**: Implementation of advanced architectures including interaction heads and relationship detection
- **Multiple Dataset Support**: HICO-DET, V-COCO, and custom dataset formats
- **Comprehensive Evaluation**: mAP, Recall@K, relationship accuracy metrics
- **Interactive Demo**: Streamlit/Gradio interface for real-time HOI detection
- **Production Ready**: Clean code structure, type hints, comprehensive testing

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Human-Object-Interaction-Detection.git
cd Human-Object-Interaction-Detection

# Install dependencies
pip install -e .

# Install optional dependencies for development
pip install -e ".[dev,detectron2]"
```

### Basic Usage

```python
from hoi_detection import HOIDetector
import cv2

# Initialize detector
detector = HOIDetector()

# Load image
image = cv2.imread("path/to/image.jpg")

# Detect HOI
results = detector.predict(image)

# Visualize results
detector.visualize(image, results)
```

### Demo

Launch the interactive demo:

```bash
streamlit run demo/app.py
```

## Dataset Schema

### HICO-DET Format
```
data/
├── hico_20160224_det/
│   ├── images/
│   │   ├── train2015/
│   │   └── test2015/
│   ├── annotations/
│   │   ├── trainval_hico.json
│   │   └── test_hico.json
│   └── hico_list_vb.txt
```

### V-COCO Format
```
data/
├── v-coco/
│   ├── images/
│   │   ├── train2014/
│   │   ├── val2014/
│   │   └── test2014/
│   ├── annotations/
│   │   ├── instances_vcoco_train.json
│   │   ├── instances_vcoco_val.json
│   │   └── instances_vcoco_test.json
│   └── data/
│       └── vcoco/
```

## Training

### Single GPU Training
```bash
python scripts/train.py --config configs/hico_det.yaml --data_root data/hico_20160224_det
```

### Multi-GPU Training
```bash
python scripts/train.py --config configs/hico_det.yaml --data_root data/hico_20160224_det --num_gpus 4
```

## Evaluation

```bash
python scripts/eval.py --config configs/hico_det.yaml --checkpoint checkpoints/best_model.pth --data_root data/hico_20160224_det
```

## Model Architecture

The project implements several HOI detection architectures:

1. **Baseline**: Object detection + interaction classification
2. **Advanced**: Graph-based relationship modeling with attention mechanisms
3. **State-of-the-art**: Transformer-based interaction detection

## Metrics

- **mAP**: Mean Average Precision for HOI detection
- **Recall@K**: Recall at different K values
- **Relationship Accuracy**: Accuracy of interaction classification
- **Efficiency**: FPS, MACs, parameters, VRAM usage

## Configuration

Configuration files are located in `configs/` directory:

- `hico_det.yaml`: HICO-DET dataset configuration
- `vcoco.yaml`: V-COCO dataset configuration
- `model_configs/`: Model-specific configurations

## Development

### Code Style
```bash
# Format code
black src/ tests/

# Lint code
ruff src/ tests/

# Run tests
pytest tests/
```

### Pre-commit Hooks
```bash
pre-commit install
```

## Limitations

- Requires GPU for optimal performance
- Large model sizes may require significant VRAM
- Training on full datasets requires substantial computational resources

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{hoi-detection,
  title={Advanced Human-Object Interaction Detection},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Human-Object-Interaction-Detection}
}
```

## License

MIT License - see LICENSE file for details.
# Human-Object-Interaction-Detection
