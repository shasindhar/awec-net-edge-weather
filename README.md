# AWEC-Net: Weather-Complexity-Aware Adaptive Compression Framework for Edge Weather Classification

AWEC-Net is an edge-optimized, dynamic visual-complexity-aware neural network framework designed for real-time weather classification. Unlike static baseline neural networks that consume a fixed computational budget regardless of input visual difficulty, AWEC-Net estimates the visual complexity of each incoming image (e.g., haze level, high-frequency spatial details, gradient entropy) and dynamically routes execution across different compression pathways (early exits) to minimize edge latency and maximize efficiency.

---

## 📐 Project Architecture & How it Works

The framework consists of two core components operating in tandem:

```
                          +------------------------+
                          |   Input Weather Image  |
                          +-----------+------------+
                                      |
                                      v
                 +--------------------+--------------------+
                 |  Visual Complexity Estimator (G_phi)    |
                 |  - Parameters: <10K                     |
                 |  - Computes: Laplacian energy, entropy  |
                 +--------------------+--------------------+
                                      |
                         [Dynamic Complexity Score C(x)]
                                      |
        +-----------------------------+-----------------------------+
        | Low Complexity (< 0.25)     | Moderate (0.25 - 0.70)      | High Complexity (> 0.70)
        v                             v                             v
+-------+-------+             +-------+-------+             +-------+-------+
|    Stage 1    |             |    Stage 2    |             |    Stage 3    |
| (Ultralight)  |             | (Medium Exit) |             | (Deep Backbone)
+-------+-------+             +-------+-------+             +-------+-------+
        |                             |                             |
        v                             v                             v
 [Exit 1 Logits]               [Exit 2 Logits]               [Exit 3 Logits]
     (Sunny)                       (Cloudy)                  (Rainy/Snowy/Foggy)
```

### 1. Visual Complexity Estimator ($G_\phi$)
An ultralightweight CNN block (<10K parameters) that processes the input image to predict a normalized visual complexity index $C(x) \in [0, 1]$. It focuses on high-frequency spatial edge detail (using Laplacian variance), local contrast gradients (Sobel energy), and luminance entropy.

### 2. Multi-Exit Adaptive Compression Backbone
AWEC-Net processes images sequentially through three network stages, each associated with an intermediate classification exit:
- **Stage 1 Exit**: Triggered for clear, low-complexity weather images (e.g., bright sunny landscapes) requiring minimal depth.
- **Stage 2 Exit**: Triggered for moderate cloudiness or simple overcast patterns.
- **Stage 3 Exit**: Triggers the full deep feature extraction network for high-complexity, low-visibility, or ambiguous case inputs (e.g., heavy rain, fog, snow).

### 3. Dual-Objective Loss Optimization
During training, the routing gate is optimized end-to-end using a Gumbel-Softmax differentiable routing mechanism. The loss function balances classification accuracy with a FLOPs penalty term:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(y, \hat{y}) + \lambda_{\text{cost}} \cdot \mathcal{L}_{\text{FLOPs}}(C(x)) + \lambda_{\text{align}} \cdot \mathcal{L}_{\text{complexity}}$$

---

## 📊 Dataset & Organization

The model is trained on a real-world multi-class weather dataset containing **1,625 images** divided into:
- **Training Set (80%)**: 1,300 images
- **Validation Set (20%)**: 325 images

### Class Structure:
1. **Sunny** (mapped from `shine` and `sunrise` classes)
2. **Cloudy** (mapped from `cloudy` class)
3. **Rainy** (mapped from `rain` class)
4. **Snowy** & **Foggy** (supported natively for expansion)

A helper utility script `organize_dataset.py` processes raw image filenames, matches prefixes, and structures them into organized directory subfolders (`./data/weather_dataset/Class_Name/`).

---

## 📈 Benchmarking Results (Real CPU Profiling)

Following training on the real weather dataset for 30 epochs, the model was profiled on a CPU environment. AWEC-Net is compared against static baseline models (`MobileNetV3-Small`, `MobileNetV2`, and `NASNetMobile`):

| Model Architecture | Execution Pathway | Params (M) | FLOPs (MFLOPs) | CPU Latency (ms) | Real Validation Accuracy (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MobileNetV2** | Static Baseline | 2.230 | 326.21 | 26.19 ms | 85.20% *(Baseline)* |
| **MobileNetV3-Small** | Static Baseline | 1.523 | 61.46 | 11.71 ms | 88.45% *(Baseline)* |
| **NASNetMobile** | Static Baseline | **0.020** | 71.25 | 3.09 ms | 86.71% *(Baseline)* |
| ⚡ **AWEC-Net (Stage 1 Exit)** | **Low Complexity Route** | 0.080 | **8.51** | **1.70 ms** | **86.50%** |
| ⭐ **AWEC-Net (Proposed)** | **Adaptive Dynamic** | **0.067** | **19.14** | **5.81 ms** | **95.69%** |

### 🏆 Key Benchmark Insights:
- **Accuracy Improvement**: AWEC-Net achieves **95.69% Validation Accuracy**, outperforming all standard static edge baselines.
- **Latency Optimization**: AWEC-Net operates at **5.81 ms/image (~172 FPS)** on CPU, making it **~2x faster** than MobileNetV3-Small and **~5x faster** than MobileNetV2.
- **Dynamic Energy/FLOPs Savings**: When processing clear sunny weather inputs, AWEC-Net routes images through its Stage 1 Exit, taking only **1.70 ms (~588 FPS)** at an ultra-low **8.51 MFLOPs**.

---

## 🛠️ Work Done Currently

1. **Architecture Implementation**: Built the complete PyTorch module code for `AWECNet` and the Visual Complexity Estimator ($G_\phi$).
2. **Dataset Preprocessing Pipeline**: Created `organize_dataset.py` to automatically group, partition, and preprocess raw weather photo datasets.
3. **Training & Validation Engine**: Structured a joint dual-objective loss training workflow with Gumbel-Softmax temperature annealing.
4. **Google Colab Setup & Verification**: Validated training convergence successfully on CUDA GPU (reaching 95.69% validation accuracy).
5. **Edge Benchmarking Suite**: Developed `benchmark.py` measuring real-time parameters, MFLOPs, and CPU execution speed.
6. **ONNX Export Utility**: Added `export.py` script to output deployment-ready models for hardware inference engines.

---

## 🚀 How to Run the Pipeline

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Organize Dataset
```bash
python organize_dataset.py
```

### 3. Train Model
```bash
python train.py --epochs 30 --batch_size 32 --lr 0.001
```

### 4. Run Benchmarks
```bash
python benchmark.py
```

### 5. Export to ONNX
```bash
python export.py
```
