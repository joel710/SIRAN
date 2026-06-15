
---


# Siran

Advanced, ultra-lightweight on-device computer vision engine purpose-built for real-time NSFW detection and non-consensual explicit content prevention.

## Overview

Siran is the core content moderation subsystem developed for the Echo social network. Engineered specifically for high-efficiency deployment in resource-constrained environments, it utilizes an optimized MobileNetV3 architecture to intercept and analyze explicit imagery before server ingestion. 

A primary engineering objective of Siran is demographic equity. Through targeted dataset balancing and aggressive color-space augmentation, the model mitigates historical algorithmic biases associated with dark skin tones, ensuring uniform precision across diverse human phenotypes.

### Key Architectural Pillars

*   **Zero-Latency On-Device Inference:** Quantized execution profile designed to run client-side within mobile applications without blocking the main UI thread.
*   **Ethical Data Augmentation:** Fine-tuned to dissociate melanin density from structural nudity, eliminating false-positive triggers common in standard open-source models.
*   **Privacy-First Design:** Content processing occurs locally within the application sandbox. Zero image data is transmitted to third-party APIs or external validation servers.

---

## Technical Specifications

The engine leverages a customized MobileNetV3-Small backbone, stripping the standard ImageNet classification head and replacing it with a specialized binary classification topology.

| Metric | Specification |
| :--- | :--- |
| **Model Architecture** | MobileNetV3-Small (Modified) |
| **Input Tensor Dimension** | 224 x 224 x 3 (RGB) |
| **Disk Footprint (INT8 Quantized)** | ~3.4 MB |
| **Memory Footprint (RAM)** | < 12 MB during active inference |
| **Average Inference Speed (CPU)** | ~15ms (Mid-range ARMv8 architecture) |
| **Target Framework** | TensorFlow Lite / ONNX Mobile Runtime |

---

## Mitigating Demographic Bias

Standard public NSFW datasets exhibit significant skew toward Western demographic distributions. When deployed in African ecosystems, these models frequently generate false positives due to a failure to distinguish between shadow gradients, high-contrast environments, and actual skin exposure.

SIRAN resolves this through a two-pronged training methodology:

1.  **Phenotypic Stratification:** Integration of the FairFace and Casual Conversations v2 datasets into the Safe-For-Work (SFW) training partition, forcing the network to learn diverse facial and anatomical representations under varying ambient lighting conditions.
2.  **Color Jittering & Stochastic Grayscale:** During the optimization phase, images undergo random brightness ($[0.6, 1.4]$), contrast ($[0.6, 1.4]$), and saturation ($[0.8, 1.2]$) modulations, alongside a 20% probability of complete serialization to grayscale. This teaches the convolutional layers to prioritize structural geometry, edge boundaries, and contextual textures over absolute color space values.

---

## Repository Structure

```text
SIRAN/
├── .github/
│   └── workflows/              # CI/CD pipeline definitions
├── core/
│   ├── __init__.py
│   ├── model.py                # MobileNetV3-Small with binary classification head
│   ├── transforms.py           # Bias-mitigation augmentation (ColorJitter, grayscale)
│   └── train.py                # Full training loop (BCEWithLogits, AdamW, CosineAnnealing)
├── dataset/
│   ├── __init__.py
│   └── prepare.py              # Raw image ingestion, train/val split, directory structuring
├── export/
│   ├── __init__.py
│   └── convert_to_tflite.py    # PyTorch → ONNX → INT8 quantized TFLite conversion
├── tests/
│   ├── __init__.py
│   ├── test_model.py           # Model output shape and range validation
│   └── test_transforms.py      # Augmentation pipeline correctness tests
├── .gitignore
├── requirements.txt            # Python dependency manifest
├── info.md                     # Training data strategy & bias mitigation reference
└── README.md
```

---

## Getting Started

### Prerequisites

* Python 3.10 or higher
* PyTorch 2.0+
* CUDA Toolkit (Optional, for accelerated training)

### Installation

Clone the repository and install the required dependencies within a virtual environment:

```bash
git clone [https://github.com/your-organization/Siran.git](https://github.com/your-organization/Siran.git)
cd Siran
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

### Model Fine-Tuning

To execute the bias-mitigated training pipeline:

```bash
python core/train.py \
  --dataset_path ./data/processed \
  --epochs 10 \
  --batch_size 32 \
  --lr 0.001 \
  --output_dir ./models/checkpoints

```

### Dataset Preparation

Structure raw images into a train/val split:

```bash
python dataset/prepare.py \
  --input_dir ./data/raw \
  --output_dir ./data/processed \
  --val_ratio 0.15
```

### Export and Quantization

Export to ONNX with INT8 quantization:

```bash
python export/convert_to_tflite.py \
  --checkpoint ./models/checkpoints/best_model.pt \
  --format onnx \
  --quantize \
  --output ./export/siran_v1
```

Or convert to TFLite for mobile deployment:

```bash
python export/convert_to_tflite.py \
  --checkpoint ./models/checkpoints/best_model.pt \
  --format tflite \
  --output ./export/siran_v1
```

---

## Deployment Strategy

The generated `.tflite` file is designed to be fetched dynamically by the client application during background synchronization sequences. This minimizes initial application binary overhead while ensuring local storage persistence.

### Integration Fallback Matrix

```
[Image Selected] ──► [Is SIRAN Local?] ──► YES ──► Local Inference (Zero Cost)
                           │
                           └──► NO ──► Secure API Ingestion ──► Async Model Download

```

---


## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
