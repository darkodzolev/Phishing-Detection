# Phishing Email Detection (NLP Project)

A Natural Language Processing project for classifying emails as **phishing (spam)** or **legitimate**.  
Two approaches are implemented and compared: a **Naive Bayes baseline** and a **deep learning LSTM model**.

---

## Dataset

Phishing Email Dataset (Kaggle)  
Final subset used: **39,154 emails**  
- 21,842 phishing (label 1)  
- 17,312 legitimate (label 0)  
- Columns include: sender, receiver, date, subject, body, label  

The **email body** is used as the input text.

---

## Preprocessing

- Lowercasing  
- Removing URLs, HTML, punctuation, numbers  
- Removing extra whitespace  
- Stopwords (for LSTM)  
- Saved as `data/processed/cleaned.csv`

---

## 🧪 Models

### **1) Naive Bayes (TF–IDF baseline)**  
- 10,000 TF–IDF features  
- `MultinomialNB` classifier  
- Model saved as `naive_bayes.pkl`  

**Results:**  
- Accuracy: ~**0.97**  
- Legitimate: Precision 0.94 / Recall 1.00  
- Phishing: Precision 1.00 / Recall 0.95  

---

### **2) LSTM (Deep Learning)**  
- Tokenizer (10k vocab)  
- Sequences padded to length 300  
- Embedding → LSTM → Dense  
- Model saved as `lstm.keras`  

**Results:**  
- Accuracy: ~**0.99**  
- Both classes: ~0.99 precision/recall/F1  

---

## Summary

- Naive Bayes provides a strong baseline (~97% accuracy).  
- LSTM improves performance to ~99%, capturing deeper linguistic patterns.  
- Both models show excellent performance on phishing detection.

---

## Notebooks

- **01** — Exploration  
- **02** — Preprocessing  
- **03** — Naive Bayes  
- **04** — LSTM  

---

## Notes

- TensorFlow requires Python 3.10/3.11 (used in this project).  
- All results and models are stored in `/results` and `/models`.