# Music Source Separation (MSS) using Band-Split RoFormer

A State-of-the-Art (SOTA) implementation of **Band-Split RoFormer** for Music Source Separation, designed to separate stereo audio into 4 stems: **Vocals, Drums, Bass, and Other**.

This project implements the architecture described in recent literature, utilizing band-split feature extraction, a Rotary Transformer (RoFormer) backbone, and multi-domain loss functions (Frequency L1 + Phase + SISDR) to achieve high-fidelity separation.

## ✨ Features

*   **Band-Split RoFormer Architecture**: Efficient frequency-band feature processing with Transformer encodings.
*   **Multi-Domain Loss**: Combines L1 Magnitude Loss, Complex/Phase Loss, and Time-Domain SISDR (Scale-Invariant Source-to-Distortion Ratio) for robust training.
*   **MUSDB18 Support**: Native integration with the MUSDB18 dataset (and MUSDB18-HQ).
*   **Robust Training Pipeline**: Supports gradient accumulation, checkpoint resuming, learning rate warmup, and validation logging.
*   **Inference & Demo**: CLI tools for separating audio files and a Gradio web interface for easy interaction.
*   **Evaluation**: Built-in evaluation script using `museval` to calculate SDR metrics.

## 📊 Current Training Status
*As of Jan 4, 2026*

The model is currently training and has resumed from **Epoch 11**.
*   **Current Epoch**: 12 / 100
*   **Progress**: ~8% of Epoch 12 completed
*   **Loss**: ~0.826 (Fluctuating)
*   **SISDR**: ~2.32 dB (Fluctuating)

*Training is active and logs are being streamed to `resume.log`.*

---

## 🛠️ Installation

### Prerequisites
*   Python 3.8+
*   NVIDIA GPU with CUDA support (Recommended)

### Setup

1.  **Clone or Open the Project**:
    Ensure you are in the project root directory.

2.  **Create and Activate Virtual Environment**:
    ```bash
    # Create
    python -m venv venv

    # Activate (Windows)
    .\venv\Scripts\activate

    # Activate (Linux/Mac)
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Verify GPU**:
    Run the included script to check if Pytorch detects your GPU.
    ```bash
    python check_gpu.py
    ```

---

## 📂 Data Preparation

This project uses the **MUSDB18** dataset.

1.  **Download MUSDB18**:
    *   If you have the full dataset, set the `MUSDB_PATH` environment variable or pass the root path explicitly.
    *   For testing/development, the code supports the 7-second snippet version (`musdb-hq` integration).

2.  **Configuration**:
    The dataset loader (`src/dataset.py`) handles splitting into Train/Validation sets and performing random mixing/augmentation during training.

---

## 🚀 Usage

### 1. Training

To train the model from scratch or resume from a checkpoint:

```bash
python -m src.train --root "C:\Path\To\MUSDB18" --epochs 100 --batch_size 2
```

**Common Arguments:**
*   `--root`: Path to MUSDB18 dataset root.
*   `--output_dir`: Directory to save checkpoints (default: `models/checkpoints`).
*   `--resume_checkpoint`: Path to a `.pt` file to resume training from.
*   `--accumulation_steps`: Gradient accumulation steps to simulate larger batch sizes (e.g., 16).
*   `--dim`, `--depth`, `--heads`: Model hyperparameters.

**Example (Resume):**
```bash
python -m src.train --root "path/to/musdb" --resume_checkpoint "models/checkpoints/checkpoint_ep10.pt"
```

### 2. Inference (Separation)

Separate a music file into 4 stems:

```bash
python src/inference.py "path/to/song.mp3" --checkpoint "models/checkpoints_optimized/checkpoint_ep11.pt" --output_dir "outputs/separated"
```

**Arguments:**
*   `--dim`, `--depth`, `--heads`: Must match the training configuration (Default: dim=128, depth=4, heads=4).

### 3. Evaluation

Evaluate the model on the MUSDB18 Test set using standard metrics (SDR, SIR, SAR):

```bash
python src/evaluate.py --checkpoint "models/checkpoints/best_model.pt" --root "path/to/musdb"
```

### 4. Web Demo (Gradio)

Launch a local web interface to upload and separate tracks:

```bash
python src/app.py
```
*Note: You may need to edit `src/app.py` to point to your specific checkpoint file.*

---

## 🏗️ Project Structure

```text
.
├── src/
│   ├── app.py           # Gradio Web Demo
│   ├── dataset.py       # MUSDB18 Dataset & DataLoader
│   ├── evaluate.py      # Museval Evaluation Script
│   ├── inference.py     # Inference/Separation Script
│   ├── layers.py        # Custom Transformer & Band-Split Layers
│   ├── loss.py          # Multi-Domain Loss Functions
│   ├── model.py         # Band-Split RoFormer Architecture
│   ├── train.py         # Main Training Loop
│   └── utils.py         # STFT/iSTFT and Audio Utilities
├── models/
│   └── checkpoints/     # Saved Model Weights
├── outputs/             # Separation Results
├── requirements.txt     # Python Dependencies
├── check_gpu.py         # Utility to check CUDA
└── README.md            # Project Documentation
```

## 🔬 Model Architecture

The **Band-Split RoFormer** works by:
1.  **Band-Split**: Dividing the spectrogram into sub-bands (Frequency intervals).
2.  **Transformer (RoPE)**: Processing these bands across time using a Rotary Positional Embedding Transformer to capture temporal dependencies.
3.  **Band-Merge**: Reconstructing the mask for each stem from the processed embeddings.
4.  **Masking**: Applying the predicted complex masks to the input mixture spectrogram to isolate each source.

---

## 📜 License
[Your License Here]
