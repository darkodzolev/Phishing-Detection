# Phishing-Detection

A Natural Language Processing project for classifying emails as phishing or legitimate using a classic ML baseline (Naive Bayes) and a deep learning model (LSTM).

Important Files
- Exploration: [`notebooks/01_exploration.ipynb`](notebooks/01_exploration.ipynb)
- Preprocessing: [`notebooks/02_preprocessing.ipynb`](notebooks/02_preprocessing.ipynb)
- Naive Bayes Model: [`notebooks/03_naive_bayes.ipynb`](notebooks/03_naive_bayes.ipynb)
- LSTM Model: [`notebooks/04_lstm.ipynb`](notebooks/04_lstm.ipynb)
- Model Comparison: [`notebooks/05_comparison.ipynb`](notebooks/05_comparison.ipynb)
- Saved Models: [`models/naive_bayes.pkl`](models/naive_bayes.pkl), [`models/lstm.h5`](models/lstm.h5)
- Evaluation Reports: [`results/nb_report.txt`](results/nb_report.txt), [`results/lstm_report.txt`](results/lstm_report.txt)
- Projektreport: [`report/Report.md`](report/Report.md)
- Dependencies: [`requirements.txt`](requirements.txt)

## 1) Setup

Open a terminal inside the project folder and create the environment:
```
cd "C:\path\to\PHISHING-EMAIL-NLP"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Start Jupyter Notebook:
```
jupyter notebook
``` 

## 2) Prepare the Data

Dataset: Phishing Email Dataset (Kaggle)

Stored at: [`data/raw/CEAS_08.csv`](data/raw/CEAS_08.csv)

After preprocessing: [`data/processed/cleaned.csv`](data/processed/cleaned.csv)

- `clean_body` (text after cleaning)
- `label` (0 = legitimate, 1 = phishing)

## 3) Dataset Exploration

Notebook: [`01_exploration.ipynb`](notebooks/01_exploration.ipynb)

Includes:
- Shape, columns, dtypes
- Missing value inspection
- Label distribution (bar plot)
- Email body length statistics
- Histogram of text length

## 4) Preprocessing

Notebook: [`02_preprocessing.ipynb`](notebooks/02_preprocessing.ipynb)

Includes:
- Lowercasing
- Remove URLs, HTML, numbers, punctuation
- Normalize whitespace
- Create `clean_body`
- Save cleaned dataset

## 5) Train the Models

### Naive Bayes (Baseline)

Notebook: [`03_naive_bayes.ipynb`](notebooks/03_naive_bayes.ipynb)

Process:
- TF-IDF with 10,000 features
- Stratified train/test split
- Train `MultinomialNB`
- Classification report + confusion matrix
- Save model: `models/naive_bayes.pkl`

Expected performance:
- Accuracy: ~0.97
- F1-score: ~0.97-0.98

### LSTM (Deep Learning)

Notebook: [`04_lstm.ipynb`](notebooks/04_lstm.ipynb)

Process:
- Tokenization (vocab size 10,000)
- Padding to length 300
- Embedding → LSTM → Dense
- Validation split during training
- Accuracy curves
- Classification report + confusion matrix
- Save model: `models/lstm.h5`

Expected performance:
- Accuracy: ~0.98-0.99
- F1-score: ~0.98-0.99

## 6) Model Comparison

Notebook: [`05_comparison.ipynb`](notebooks/05_comparison.ipynb)

Includes:
- Parsing both classification reports
- Comparison tables (accuracy, macro/weighted F1)
- Bar plots: overall metrics
- Bar plots: class-wise F1
- Final summary of results

The LSTM model outperforms Naive Bayes by about 1 percentage point in overall accuracy and macro F1.

## 7) File Descriptions

- `01_exploration.ipynb`: Initial dataset inspection and basic statistics

- `02_preprocessing.ipynb`: Text cleaning and export

- `03_naive_bayes.ipynb`: Baseline model with TF-IDF + NB

- `04_lstm.ipynb`: Deep learning model (embedding + LSTM)

- `05_comparison.ipynb`: Full side-by-side evaluation of NB and LSTM

- `models/`: Saved model artifacts

- `results/`: Text-based classification reports (sklearn output)

## 8) Common Issues

- TensorFlow errors: ensure Python 3.10 or 3.11
- Import errors (`ModuleNotFoundError`): activate the virtual environment before starting Jupyter:

    ```
    .venv\Scripts\activate
    jupyter notebook
    ```

## 9) Notes

- Notebook 05 is ideal for presentation (plots + summary).
- Naive Bayes is extremely fast and strong for this dataset.
- LSTM performs best and captures deeper patterns.
- Entire workflow is reproducible through the notebooks.