# Support Ticket Classification & Priority Prediction System

Production-style NLP project for classifying customer support tickets and predicting operational priority.

The system takes raw ticket text and returns:

- Ticket category
- ML-predicted priority
- Rule-boosted final priority
- Detected escalation keywords
- Top TF-IDF terms for basic interpretability

Example:

```text
Input: Payment deducted twice and refund not received
Output: Category = Billing/related ticket type, Priority = High, Boosted Priority = High
```

## Business Problem

Support teams receive large volumes of unstructured tickets. Manual triage can delay urgent cases, create inconsistent routing, and make it harder to measure operations. This project demonstrates a realistic NLP triage workflow that can route tickets by category and escalate priority when business-critical language appears.

## Dataset

Kaggle dataset: Customer IT Support - Ticket Dataset  
https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets

Expected local file:

```text
data/raw/dataset-tickets-german_normalized_50_5_2.csv
```

Raw and processed data files are ignored by Git. The folder structure is tracked with `.gitkeep` files.

## Project Structure

```text
support-ticket-classification/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── category_model.pkl
│   ├── priority_model.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_category_classification.ipynb
│   ├── 04_priority_prediction.ipynb
│   └── 05_model_evaluation.ipynb
├── outputs/
├── src/
│   ├── preprocessing.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Architecture

```text
Raw Ticket Text
      |
      v
NLTK Cleaning: lowercase, punctuation removal, tokenization, stopword removal, lemmatization
      |
      v
TF-IDF Vectorizer: max_features, min_df, unigrams + bigrams
      |
      +--------------------------+
      |                          |
      v                          v
Category Classifier         Priority Classifier
LogReg / LinearSVC / NB     LogReg / LinearSVC
      |                          |
      v                          v
Predicted Category          ML Priority
                                 |
                                 v
Keyword Escalation Rules -> Final Priority
```

Two separate models are used intentionally:

- `ticket_text -> category`
- `ticket_text -> priority`

This is easier to debug, evaluate, retrain, and explain than a single multitask model.

## Why TF-IDF?

TF-IDF converts text into sparse numeric features while giving higher weight to terms that are important in a ticket but not common across every ticket. It is simple, fast, strong for classical NLP baselines, and appropriate for internship-quality production prototypes.

The vectorizer uses:

- `max_features=12000`
- `ngram_range=(1, 2)`
- `min_df=2`
- `sublinear_tf=True`

## Why LinearSVC Works Well For Text

Support ticket TF-IDF matrices are high-dimensional and sparse. Linear SVMs often perform well in this setting because they learn strong separating hyperplanes efficiently, handle many features, and are less likely to overfit individual noisy words than more flexible models.

## Hybrid Priority Boosting

The project includes business rules for escalation keywords such as:

```text
refund, hacked, urgent, payment failed, server down, cannot access, breach
```

If the ML model predicts `Medium` but a critical keyword is detected, the final priority is boosted to `High`. Real businesses use hybrid ML plus rules because some operational risks should be escalated deterministically, even when the model is uncertain.

## Why Accuracy Alone Is Insufficient

Accuracy can hide poor performance on minority classes. In support operations, missing rare but urgent tickets is costly. This project reports accuracy, macro precision, macro recall, macro F1-score, classification reports, and confusion matrices.

Confusion matrices matter because they show exactly which categories or priorities are being confused, which helps improve routing logic and training data quality.

## How To Run

Install dependencies:

```powershell
pip install -r requirements.txt
```

Train models:

```powershell
python src/train.py
```

This creates:

```text
models/category_model.pkl
models/priority_model.pkl
models/tfidf_vectorizer.pkl
data/processed/processed_tickets.csv
outputs/category_model_comparison.json
outputs/priority_model_comparison.json
outputs/category_confusion_matrix.png
outputs/priority_confusion_matrix.png
```

Run a prediction:

```powershell
python src/predict.py
```

Launch the demo app:

```powershell
streamlit run app/streamlit_app.py
```

## Model Comparison

Category classification compares:

- Logistic Regression
- LinearSVC
- Multinomial Naive Bayes

Priority prediction compares:

- Logistic Regression
- LinearSVC

The training script selects the best model by macro F1-score and saves it.

Current local training run:

```text
Rows processed: 13178
Best category model: LinearSVC
Category accuracy: 0.9738
Category macro F1: 0.9738
Best priority model: LinearSVC
Priority accuracy: 0.9636
Priority macro F1: 0.9607
```

These scores are intentionally reported with both accuracy and macro F1 because class balance matters in support operations. The stronger results come from using a cleaner ticket dataset with better text-label alignment.

## Results And Screenshots

Generated evaluation assets are saved in `outputs/`:

- `category_distribution.png`
- `priority_distribution.png`
- `ticket_length_distribution.png`
- `category_confusion_matrix.png`
- `priority_confusion_matrix.png`
- `category_model_comparison.json`
- `priority_model_comparison.json`
- `training_summary.json`

Add Streamlit screenshots here after running:

```powershell
streamlit run app/streamlit_app.py
```

## Interpretability

The prediction pipeline returns the strongest TF-IDF terms in the ticket. This provides a simple explanation of which words had the most signal in the vectorized input.

Example:

```python
{
    "category": "Billing",
    "priority": "Medium",
    "boosted_priority": "High",
    "detected_keywords": ["refund"],
    "top_terms": ["refund", "payment", "charged"]
}
```

## Future Improvements

- Add semantic similarity search for historical tickets
- Add calibrated confidence scores for LinearSVC
- Add cross-validation reports
- Add misclassification analysis notebook
- Add class imbalance experiments with resampling
- Deploy the Streamlit app to Streamlit Community Cloud
