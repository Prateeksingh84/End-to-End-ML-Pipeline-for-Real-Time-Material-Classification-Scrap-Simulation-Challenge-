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

```
.
├── .vscode/                   # VS Code configurations
├── data/                      # Dataset
│   ├── cardboard/             
│   ├── glass/                 
│   ├── metal/                 
│   ├── paper/                 
│   ├── plastic/               
│   └── test_frames/           # Test images for simulation
├── models/                    
│   ├── trained_model.pth      # Trained PyTorch checkpoint
│   └── converted_model.onnx   # ONNX-format model for deployment
├── results/                   
│   └── simulation_results.csv # Conveyor simulation logs
├── src/                       
│   ├── data_preparation.py    # Data loading & augmentation
│   ├── model_training.py      # Model training & evaluation
│   ├── inference_script.py    # Inference + ONNX conversion
│   └── simulation.py          # Real-time conveyor simulation
├── performance_report.md      # Model performance summary
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

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
```

### 🔹 2. Train the Model  
```bash
python src/model_training.py
```

### 🔹 3. Convert to ONNX & Run Inference  
```bash
python src/inference_script.py
```

### 🔹 4. Run Real-Time Simulation  
```bash
python src/simulation.py
```

📂 Results logged at:  
```
results/simulation_results.csv
```

---

## 📑 Reports  

- **performance_report.md** → Model evaluation  
- **simulation_results.csv** → Conveyor predictions log  

---

## 🏆 Key Highlights  

- End-to-end ML pipeline (**data → training → deployment → simulation**)  
- Lightweight **ResNet18** with transfer learning  
- **ONNX-ready** for cross-platform real-time inference  
- Modular & extensible codebase  

---

## 🔮 Future Work  

- Expand dataset with more categories  
- Switch from **classification → object detection**  
- Deploy pipeline on **edge devices** (Jetson Nano, Raspberry Pi)  
- Optimize inference via **quantization & pruning**  

---

## 👨‍💻 Author  

**Prateek Singh**  
📅 Project: **Scrap Simulation Challenge**  

---

## 📜 License  

This project is licensed under the **MIT License**.  
