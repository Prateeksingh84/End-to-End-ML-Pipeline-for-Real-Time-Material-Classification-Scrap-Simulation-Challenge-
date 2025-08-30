# 🏭 Scrap Simulation Challenge – End-to-End ML Pipeline for Real-Time Material Classification

This project implements an **end-to-end machine learning pipeline** to classify scrap materials from image data, simulating a **real-time scrap sorting conveyor**.  
It demonstrates proficiency in **data processing, model development, deployment, and documentation**.

---

## 📌 1. Project Objective
The primary objective is to build a **robust, mini-classification pipeline** that:
- Trains on image data of different scrap materials  
- Deploys a **lightweight model** for inference  
- Simulates a **real-time scrap sorting system**  

---

## 📂 2. Project Deliverables & Folder Structure
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

## 📊 3. Dataset
- Dataset inspired by **TrashNet**  
- Contains at least **5 classes of scrap materials**:  
  - Cardboard  
  - Glass  
  - Metal  
  - Paper  
  - Plastic  

✅ Pre-labeled structure makes training easier.  
✅ Direct relevance to real-world recycling scenarios.  

---

## 🧠 4. Model Architecture & Training

### 🔹 Architecture
- Base model: **ResNet18 (CNN)**
- Chosen for its **balance of accuracy and efficiency**
- Lightweight enough for **real-time deployment**

### 🔹 Training Process
1. **Preprocessing**  
   - Images resized to `224x224`  
   - Normalized pixel values  

2. **Data Augmentation**  
   - Random horizontal flip  
   - Color jitter  
   - Boosts generalization  

3. **Transfer Learning**  
   - Initialized with **ImageNet pretrained weights**  
   - Faster convergence, less data needed  

4. **Fine-tuning**  
   - Replaced and retrained final FC layer  
   - Convolutional base frozen  

5. **Evaluation Metrics**  
   - Accuracy  
   - Precision & Recall  
   - Confusion Matrix  

---

## 🚀 5. Deployment Decisions
- Trained model saved as **PyTorch checkpoint (`.pth`)**
- Converted to **ONNX format** for deployment  

**Why ONNX?**
- ✅ **Portability** – Run across platforms & devices  
- ✅ **Performance** – Optimized inference speed  
- ✅ **Scalability** – Suitable for real-time conveyor simulation  

---

## ▶️ 6. How to Run

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt

### 2️⃣ Train the Model
```bash
python src/model_training.py

### 3️⃣ Convert to ONNX & Run Inference
```bash
python src/inference_script.py

### 4️⃣ Run the Real-Time Simulation
```bash
python src/simulation.py

### Simulation results will be stored in:
```bash
results/simulation_results.csv

## 📑 7. Reports
- **performance_report.md** → Contains detailed model evaluation  
- **simulation_results.csv** → Logs all predictions during conveyor simulation  

---

## 🏆 8. Key Highlights
- End-to-end ML pipeline from **data → training → deployment → simulation**  
- **Lightweight ResNet18 model** with transfer learning  
- **ONNX deployment** for real-time performance  
- **Modular code structure** for easy extension  

---

## 🔮 Future Work
- Expand dataset with more scrap categories  
- Integrate **object detection** (instead of classification)  
- Deploy model to an **edge device (Jetson Nano / Raspberry Pi)**  
- Optimize pipeline with **quantization / pruning**  

---

## 👨‍💻 Author
- **PRATEEK**  
- 📅 **Project:** Scrap Simulation Challenge  
