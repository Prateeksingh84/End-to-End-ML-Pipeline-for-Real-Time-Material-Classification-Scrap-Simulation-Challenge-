# 🏭 Scrap Simulation Challenge  
_End-to-End ML Pipeline for Real-Time Scrap Material Classification_  

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)  
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red?logo=pytorch)  
![ONNX](https://img.shields.io/badge/ONNX-Ready-green?logo=onnx)  
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)  
![License](https://img.shields.io/badge/License-MIT-yellow)  

---

## 📌 Overview  

This project demonstrates an **end-to-end machine learning pipeline** that classifies scrap materials from image data and simulates a **real-time conveyor-based sorting system**.  

It showcases:  
- 🔹 Data processing & augmentation  
- 🔹 Model training with **ResNet18** (transfer learning)  
- 🔹 Deployment using **ONNX** for fast inference  
- 🔹 Real-time conveyor **simulation with logging**  

---

## 🎯 Objectives  

- Build a robust ML pipeline for **scrap material classification**  
- Enable **real-time inference** with ONNX  
- Simulate conveyor sorting with **live predictions**  

---

## 📂 Project Structure  
```bash
.
├── .vscode/                   # VS Code-specific configurations
├── data/                      # Contains the dataset
│   ├── [class_1]/             # e.g., metal/
│   ├── [class_2]/             # e.g., plastic/
│   └── test_frames/           # Images used in the simulation
├── models/                    # Stores the trained and converted models
│   ├── trained_model.pth      # The PyTorch model checkpoint
│   └── converted_model.onnx   # The ONNX-format model for deployment
├── results/                   # Contains the output from the simulation
│   └── simulation_results.csv # Log of all classification results
├── src/                       # All source code
│   ├── data_preparation.py    # Handles data cleaning and augmentation
│   ├── inference_script.py    # Contains the inference engine and conversion logic
│   ├── model_training.py      # Script for model training and evaluation
│   └── simulation.py          # The main script for the real-time simulation
├── .gitignore                 # Files to be ignored by Git
├── README.md                  # Project documentation
└── performance_report.md      # A summary of model performance


---

## 📊 Dataset  

Inspired by **TrashNet**, with **5 classes** of scrap materials:  
- 🟫 Cardboard  
- 🟦 Glass  
- ⚙️ Metal  
- 📄 Paper  
- 🟧 Plastic  

✅ Pre-labeled & ready for supervised training  
✅ Relevant for **real-world recycling automation**  

---

## 🧠 Model Architecture & Training  

- **Base Model:** ResNet18 (lightweight CNN)  
- **Pretrained Weights:** ImageNet  
- **Training Pipeline:**  
  - Resize to `224x224`, normalize  
  - Augmentation: horizontal flip, color jitter  
  - Replace final FC layer → retrained on scrap classes  
  - Metrics: Accuracy, Precision, Recall, Confusion Matrix  

---

## 🚀 Deployment  

- Model saved as `.pth` (PyTorch checkpoint)  
- Converted to `.onnx` for optimized inference  

**Why ONNX?**  
✔ Portability across devices  
✔ Faster inference for real-time sorting  
✔ Ready for edge deployment (Jetson Nano / Raspberry Pi)  

---

## ▶️ Usage  

### 🔹 1. Install Dependencies  
```bash
pip install -r requirements.txt
