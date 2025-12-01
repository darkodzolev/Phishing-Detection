# Phishing-Detection (Project Report)


## 1) Dataset Description

The project uses the Phishing Email Dataset (Kaggle), a combined corpus created by researchers to study phishing tactics across multiple email sources.

### Dataset Composition

The full dataset merges classic email corpora:

- Enron, Ling - content-focused datasets
    
- CEAS, Nazario, Nigerian Fraud, SpamAssassis - include metadata and spam labels
    

The merged dataset contains approximately:

- 82,500 emails
    
- 42,891 phishing/spam
    
- 39,595 legitimate
    

For this project, the subset **CEAS\_08.csv** is used. It includes:

- ~39,000 emails
    
- Columns: sender, receiver, date, subject, body, label, urls
    
- Labels: 1 = phishing, 0 = legitimate
    
- Real-world text with HTML fragments, promotional language, URLs, and scam content
    

### Why CEAS\_08 Works Well

- Large, cleanly structured text data
    
- Clear binary labels
    
- Good balance for training classical and deep models
    
- Reflects realistic variation in phishing email content
    

## 2) Relevance to Real Applications

Phishing detection is central to cybersecurity and email filtering. Real-world systems use similar pipelines to:

- Block phishing and scam emails
    
- Prevent credential theft
    
- Improve user safety in corporate inboxes
    
- Detect malicious URLs or social engineering attempts
    

### Risks & Considerations

- Dataset imbalance may bias models
    
- Pure text classification lacks header, network, and attachment features
    
- False negatives pose security risks; false positives affect legitimate email delivery
    

## 3) Issues Encountered and Solutions

### 1\. Noisy and inconsistent email formatting

Emails contained URLs, HTML tags, newline symbols, numbers, and mixed casing.

Solution: A cleaning pipeline to:

- lowercase text
    
- remove URLs and HTML
    
- remove punctuation and numbers
    
- normalize whitespace

This produced the `clean_body` column for modeling.

### 2\. Large variation in email length

Emails ranged from one line to multi-page messages.

Solution:

- TF-IDF with a fixed feature space (10,000 features)
    
- Tokenization + sequence padding to 300 tokens for LSTM
    

### 3\. TensorFlow reproducibility

Deep models produce slightly different results in each run.

Solution:

- fixed random seeds where possible
    
- keep expected performance as a range (~0.98-0.99)
    

### 4\. Class imbalance

`CEAS_08` has somewhat more phishing than legitimate examples.

Solution:

- stratified train/test split
    
- macro/weighted F1 metrics for fairness evaluation
    

## 4) Approach and Justification (What & Why)

### Preprocessing

What: Clean raw email bodies and save as `cleaned.csv`.

Why: Text noise interferes with both statistical and neural models.

### Naive Bayes Baseline

What:

- TF-IDF (10k max features)
    
- MultinomialNB classifier
    
- 80/20 stratified split
    

Why: A classic, extremely efficient baseline for email spam detection.

### LSTM Deep Learning Model

What:

- Tokenizer with 10,000 vocabulary size
    
- Fixed sequence length = 300
    
- Embedding → LSTM → Dense(sigmoid)
    
- 5 epochs, batch size 64, with validation split
    

Why: Captures sequential structure and linguistic patterns better than TF-IDF.

### Evaluation

What:

- classification reports
    
- accuracy, macro F1, weighted F1
    
- class-wise F1
    
- comparison tables and plots
    
- final performance summary
    

Why: Shows clearly how classical and neural models differ.


## 5) Results and Interpretation

### Naive Bayes

- Accuracy: ~0.97
    
- F1-score: ~0.97-0.98
    
- Handles short, keyword-heavy phishing emails well
    
- Excellent recall for legitimate emails in this subset
    

### LSTM

- Accuracy: ~0.98-0.99
    
- F1-score: ~0.98-0.99
    
- Learns deeper stylistic patterns
    
- Outperforms Naive Bayes by 1-2 percentage points
    

### Interpretation

- The dataset is highly separable: both models perform very well
    
- LSTM provides the best generalization
    
- Performance ranges match typical expectations for phishing datasets
    

## 6) Limitations and Future Work

### Limitations

- Only text is used; no header-level metadata
    
- Some emails are extremely short or noisy
    
- Using only one subset (`CEAS_08`), not the entire merged dataset
    

### Potential Extensions

- Merge all provided CSVs for a larger dataset
    
- Use transformer-based models (BERT, DistilBERT)
    
- Add engineered features (URL count, sender domain, HTML depth)
    
- Tune hyperparameters: embedding size, LSTM units, max sequence length
    
- Evaluate using more robust methods (e.g., k-fold cross-validation)
    

## 7) References

Dataset citation requested by Kaggle authors:

Al-Subaiey, A., Al-Thani, M., Alam, N. A., Antora, K. F., Khandakar, A., & Zaman, S. A. U. (2024). Novel Interpretable and Robust Web-based AI Platform for Phishing Email Detection. [arXiv:2405.11619.](https://arxiv.org/abs/2405.11619)