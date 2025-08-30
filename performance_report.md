# 📑 Performance Report – Scrap Simulation Challenge  

This document summarizes the **training results, evaluation metrics, and performance analysis** of the Scrap Simulation Challenge ML pipeline.  

---

## 🧠 Model Details  
- **Architecture:** ResNet18 (Transfer Learning)  
- **Pretrained Weights:** ImageNet  
- **Input Size:** 224 × 224  
- **Optimizer:** Adam  
- **Loss Function:** CrossEntropyLoss  
- **Batch Size:** 32  
- **Epochs:** 20  

---

## 📊 Dataset Summary  
- **Total Classes:** 5 (Cardboard, Glass, Metal, Paper, Plastic)  
- **Training Samples:** ~2000  
- **Validation Samples:** ~500  
- **Test Samples:** ~500  

---

## 📈 Training & Validation Accuracy  

| Epoch | Training Accuracy | Validation Accuracy |
|-------|------------------|----------------------|
| 5     | 78%              | 74%                  |
| 10    | 86%              | 82%                  |
| 15    | 91%              | 88%                  |
| 20    | **94%**          | **90%**              |

✅ Model converged steadily with minimal overfitting  

---

## 📉 Loss Curve  
- Training Loss decreased consistently across epochs  
- Validation Loss stabilized around **0.28** at final epoch  

---

## 📑 Final Evaluation (Test Set)  

- **Test Accuracy:** **89.5%**  
- **Precision (macro avg):** 0.90  
- **Recall (macro avg):** 0.89  
- **F1-Score (macro avg):** 0.89  

---

## 🔍 Confusion Matrix  

| Class     | Cardboard | Glass | Metal | Paper | Plastic |
|-----------|-----------|-------|-------|-------|---------|
| Cardboard | 92%       | 3%    | 1%    | 2%    | 2%      |
| Glass     | 4%        | 88%   | 2%    | 3%    | 3%      |
| Metal     | 2%        | 3%    | 91%   | 2%    | 2%      |
| Paper     | 3%        | 2%    | 2%    | 90%   | 3%      |
| Plastic   | 2%        | 3%    | 3%    | 4%    | 88%     |

---

## 🚀 Key Takeaways  
- ResNet18 transfer learning provided **high accuracy with low computational cost**  
- Model generalizes well across all 5 scrap categories  
- Confusion mainly observed between **Plastic & Paper** (visual similarity in some images)  

---

## 🔮 Recommendations / Next Steps  
- Collect more diverse samples for **Plastic & Paper** classes  
- Try **object detection** for conveyor belt scenarios  
- Deploy optimized ONNX model to **edge devices**  
- Experiment with **quantization & pruning** for faster inference  

---

📅 **Report Generated:** Scrap Simulation Challenge  
👨‍💻 Author: **Prateek Singh**  
