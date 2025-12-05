# 📰 Fake News Stance Detection using Embedding Blending, CNN, and BERT Models

This repository contains our deep learning project for **Fake News Stance Detection**, where the goal is to classify the stance of a news article body relative to its headline into four categories:

- **Agree**
- **Disagree**
- **Discuss**
- **Unrelated**

Our work builds upon a 2024 research paper that used **embedding blending + PCA + CNN**, and we extend it by introducing **DistilBERT** and **TinyBERT** for contextual learning and better generalization.

---

## 🚀 Project Overview

Fake news spreads rapidly across social media, and manual verification is slow and expensive.  
Stance detection helps identify misleading news by analyzing the *relationship* between headlines and full articles.

This project includes:

- The **base CNN model** using blended static embeddings  
- Enhanced **DistilBERT** and **TinyBERT** transformer-based models  
- Domain-shift testing for real-world generalization  
- Performance comparison across all models  

---

## 🧠 Models Implemented

### **1️⃣ Base Model – CNN + Embedding Blending**
- FastText (α = 0.33)
- FastText Subword (β = 0.33)
- GloVe (γ = 0.34)
- PCA for dimensionality reduction (95% variance)
- 1D CNN with Global Max Pooling
- Output layer: 4-class stance prediction

### **2️⃣ DistilBERT**
- Contextualized embeddings (768D)
- BERT tokenizer
- Strong improvement in accuracy & F1-score
- Better semantic understanding

### **3️⃣ TinyBERT**
- Lightweight version of BERT
- Fast inference + low memory footprint
- Suitable for real-time applications

---

## 📊 Results

| Model        | Accuracy | F1-Score | Improvement vs CNN |
|--------------|----------|----------|---------------------|
| **CNN (Base)** | 94.58%  | ~94.0%  | Baseline            |
| **DistilBERT** | 97.03%  | 96.90%  | **+2.45%** ✔        |
| **TinyBERT**   | 92.17%  | 91.10%  | −2.41%              |

DistilBERT clearly outperformed the embedding-based CNN, proving the advantage of contextual models.

---

## 📁 Repository Structure

├── 22i-1969_19i-1864_22i-1880.ipynb # Main Jupyter Notebook
├── data/ # Training & test datasets
├── models/ # Saved model weights (CNN / DistilBERT / TinyBERT)
├── results/ # Evaluation metrics & plots
├── README.md # Project documentation


---

## 🛠️ Installation & Setup

### **1. Clone the repository**
```bash
git clone https://github.com/your-username/stance-detection.git
cd stance-detection
pip install -r requirements.txt
