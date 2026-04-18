# ============================================================
# Sentiment Analysis with Multiple Models 
# ============================================================

import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Import custom modules
from src.preprocess import preprocess
from src.train_models import train_nb, train_svm
from src.evaluate import evaluate_model

# ============================================================
# 1. CREATE RESULTS FOLDER
# ============================================================

os.makedirs("results", exist_ok=True)

# ============================================================
# 2. LOAD DATASET (IMDB)
# ============================================================

print("\nLoading dataset...")

data = pd.read_csv("data/IMDB Dataset.csv")

# Rename columns
data = data.rename(columns={"review": "text", "sentiment": "label"})

# Convert labels
data['label'] = data['label'].map({'positive': 1, 'negative': 0})

# Reduce size for faster execution
data = data.sample(5000, random_state=42)

print("Dataset Loaded Successfully!")
print("Total Records:", len(data))

# ============================================================
# 3. PREPROCESSING
# ============================================================

print("\nPreprocessing Text...")

data['clean_text'] = data['text'].apply(preprocess)

print("Preprocessing Completed!")

# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    data['clean_text'], data['label'],
    test_size=0.3,
    random_state=42
)

print("\nTrain Size:", len(X_train))
print("Test Size:", len(X_test))

# ============================================================
# 5. FEATURE EXTRACTION (IMPROVED TF-IDF)
# ============================================================

print("\nApplying TF-IDF...")

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF Applied Successfully!")

# ============================================================
# 6. TRAIN MODELS
# ============================================================

print("\nTraining Models...")

nb_model = train_nb(X_train_tfidf, y_train)
svm_model = train_svm(X_train_tfidf, y_train)

print("Models Trained Successfully!")

# ============================================================
# 7. PREDICTIONS
# ============================================================

nb_pred = nb_model.predict(X_test_tfidf)
svm_pred = svm_model.predict(X_test_tfidf)

# ============================================================
# 8. EVALUATION
# ============================================================

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

evaluate_model("Naive Bayes", y_test, nb_pred)
evaluate_model("SVM", y_test, svm_pred)

# Accuracy values
nb_acc = accuracy_score(y_test, nb_pred)
svm_acc = accuracy_score(y_test, svm_pred)

# ============================================================
# 9. ABLATION STUDY
# ============================================================

print("\n==============================")
print("ABLATION STUDY")
print("==============================")

# Without preprocessing
print("\nExperiment 1: Without Preprocessing")

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    data['text'], data['label'], test_size=0.3, random_state=42
)

tfidf_raw = TfidfVectorizer(max_features=5000)
X_train_raw_vec = tfidf_raw.fit_transform(X_train_raw)
X_test_raw_vec = tfidf_raw.transform(X_test_raw)

nb_raw_model = train_nb(X_train_raw_vec, y_train_raw)
nb_raw_pred = nb_raw_model.predict(X_test_raw_vec)

print("Accuracy (No Preprocessing):",
      accuracy_score(y_test_raw, nb_raw_pred))

# CountVectorizer
print("\nExperiment 2: CountVectorizer")

count_vec = CountVectorizer(max_features=5000)
X_train_count = count_vec.fit_transform(X_train)
X_test_count = count_vec.transform(X_test)

nb_count_model = train_nb(X_train_count, y_train)
nb_count_pred = nb_count_model.predict(X_test_count)

print("Accuracy (CountVectorizer):",
      accuracy_score(y_test, nb_count_pred))

# TF-IDF comparison
print("\nExperiment 3: TF-IDF Comparison")

print("Naive Bayes (TF-IDF):", nb_acc)
print("SVM (TF-IDF):", svm_acc)

# ============================================================
# 10. GRAPH: ACCURACY COMPARISON
# ============================================================

models = ["Naive Bayes", "SVM"]
accuracy = [nb_acc, svm_acc]

plt.figure()
plt.bar(models, accuracy)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")

plt.savefig("results/accuracy_graph.png")
plt.show()

# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, svm_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix - SVM")
plt.savefig("results/confusion_matrix.png")
plt.show()

# ============================================================
# 12. SAMPLE PREDICTIONS
# ============================================================

print("\n==============================")
print("SAMPLE PREDICTIONS")
print("==============================")

sample_texts = [
    "I absolutely love this movie",
    "This is the worst film ever",
    "Not bad but could be better",
    "Amazing acting and storyline",
    "I am not satisfied with the plot"
]

sample_clean = [preprocess(text) for text in sample_texts]
sample_vec = tfidf.transform(sample_clean)

predictions = svm_model.predict(sample_vec)

for text, pred in zip(sample_texts, predictions):
    sentiment = "Positive" if pred == 1 else "Negative"
    print(f"{text} --> {sentiment}")

# ============================================================
# 13. FINAL CONCLUSION
# ============================================================

print("\n==============================")
print("FINAL CONCLUSION")
print("==============================")

print("Naive Bayes Accuracy:", nb_acc)
print("SVM Accuracy:", svm_acc)

if svm_acc > nb_acc:
    print("\nConclusion: SVM performs better on real-world dataset.")
else:
    print("\nConclusion: Naive Bayes performs better.")

print("\nExperiment Completed Successfully!")
