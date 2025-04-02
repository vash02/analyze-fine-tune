# 🔍 Parameter-Efficient Fine-Tuning & Attention Analysis for LLMs

This repository demonstrates how to **fine-tune large language models (LLMs)** using **parameter-efficient methods** like **LoRA**, **BitFit**, and **Adapters**, and then **visualize the changes** in their **attention weights**.

The project is designed to be **memory-efficient** and runs on **Apple Silicon (MPS)**, making it ideal for researchers and developers working on **resource-constrained machines**.

---

## 🚀 Features

- ✅ Fine-tuning with:
  - **LoRA** (Low-Rank Adapters)
  - **BitFit** (Bias-Only Updates)
  - **Adapter** (Selective Layer Unfreezing)
  - **Full Fine-Tuning** (for baseline comparison)
- 📉 Trains on the **Financial PhraseBank** dataset (`sentences_allagree`)
- 🔍 Extracts and compares **last-layer attention weights** across methods
- 📊 Plots **attention heatmaps** for an input sentence to analyze how fine-tuning changes model behavior
- ⚡️ Runs on **CPU or Apple MPS** (no need for CUDA!)

---

## 📂 Project Structure

```
📁 fine-tune-analyze/
│
├── fine_tune_analyze.py      # Main script to run all training and generate heatmaps
├── README.md                 # You're here!
└── /[model_checkpoints]/     # Each method’s fine-tuned model is saved here (e.g., ./phi1.5-lora/)
```

---

## 📦 Requirements

Install dependencies using:

```
pip install -r requirements.txt
```

Make sure you have the following packages installed:

- `transformers`
- `datasets`
- `peft`
- `accelerate`
- `matplotlib`
- `seaborn`
- `tqdm`

Optionally, configure `accelerate`:

```
accelerate config
```

---

## 🧪 How to Run

```
python fine_tune_analyze.py
```

This will:
- Load the dataset
- Fine-tune models with each method (or reuse saved checkpoints)
- Generate attention maps for a test sentence
- Plot and display comparisons (2×2 or 1×4 layout)

---

## 📊 Example Output

The script outputs attention heatmaps like:

- **Original Model**
- **LoRA Fine-Tuned**
- **BitFit Fine-Tuned**
- **Adapter Fine-Tuned**
- *(Optionally Full Fine-Tuned)*

These allow visual comparison of how attention weights shift after fine-tuning.

---

## 🧠 Why PEFT?

Parameter-efficient fine-tuning methods help:

- ✅ **Reduce training time and memory usage**
- ✅ **Adapt billion-parameter models to new domains**
- ✅ **Run fine-tuning on consumer hardware (M1/M2 Macs, small GPUs)**

This repo is a practical exploration of *how and why* these methods work — including what changes in the model’s internals like attention and feed-forward layers.

