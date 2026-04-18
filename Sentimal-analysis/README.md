# Sentiment Analysis with Multiple Models

## Overview  
This project implements a sentiment analysis system that classifies textual data into **positive** and **negative** sentiments using classical machine learning approaches. The system is designed to analyze real-world text data such as movie reviews and determine the emotional tone expressed in the content.  

Unlike basic implementations, this project follows a **research-oriented experimental approach**, where two widely used models—**Naive Bayes** and **Support Vector Machine (SVM)**—are compared in terms of performance, efficiency, and robustness. The system also includes preprocessing techniques, feature engineering using **TF-IDF with n-grams**, and an ablation study to evaluate the contribution of each component.  

The goal of this project is not only to build a sentiment classifier but also to understand how different techniques impact model performance in real-world scenarios.

---

## Features  

### Text Preprocessing  
- Converts text to lowercase  
- Removes punctuation and special characters  
- Tokenizes text into words  
- Removes stopwords  
- Improves overall data quality  

### Feature Extraction  
- Uses **TF-IDF (Term Frequency–Inverse Document Frequency)**  
- Assigns importance to words based on frequency  
- Supports **n-grams (unigrams and bigrams)**  
- Captures contextual patterns such as *“not good”*  

### Model Training  
- Implements **Naive Bayes classifier**  
- Implements **Support Vector Machine (SVM)**  
- Trains both models on the same dataset for comparison  

### Evaluation Metrics  
- Accuracy  
- Precision  
- Recall  
- F1-score  

### Visualization  
- Accuracy comparison graph  
- Confusion matrix  
- Helps in understanding model performance  

### Ablation Study  
- Without preprocessing  
- Using CountVectorizer  
- Using TF-IDF with Naive Bayes  
- Using TF-IDF with SVM  

---

## Output  

The system generates:  
- Model performance metrics  
- Accuracy comparison graph  
- Confusion matrix  
- Sample predictions  
- Ablation study results  

---

## Classification Criteria  

- **Positive** → Text expresses positive sentiment  
- **Negative** → Text expresses negative sentiment  

---

## Project Structure  

sentiment-analysis-multiple-models/

data/  
    dataset_large.csv  

src/  
    preprocess.py  
    train_models.py  
    evaluate.py  

results/  
    accuracy_graph.png  
    confusion_matrix.png  

main.py  
requirements.txt  
README.md  
paper.pdf  

---

## Technologies Used  

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- NLTK  
- Matplotlib  

---

## Installation  

Clone the repository:  
git clone https://github.com/your-username/sentiment-analysis-multiple-models.git  

Navigate to project folder:  
cd sentiment-analysis-multiple-models  

Install dependencies:  
pip install -r requirements.txt  

Run the project:  
python main.py  

---

## Usage  

- Load dataset  
- Preprocess text  
- Train models  
- Evaluate performance  
- Generate graphs and confusion matrix  

---

## Results  

| Model          | Accuracy | Precision | Recall | F1-score |
|---------------|---------|----------|--------|---------|
| Naive Bayes   | 0.84    | 0.83     | 0.82   | 0.82    |
| SVM           | 0.90    | 0.89     | 0.88   | 0.88    |

**Conclusion:**  
SVM performs better than Naive Bayes due to its ability to handle high-dimensional data effectively.

---

## Ablation Study Results  

| Experiment                | Accuracy |
|--------------------------|---------|
| Without preprocessing    | 0.70    |
| CountVectorizer          | 0.78    |
| TF-IDF + Naive Bayes     | 0.84    |
| TF-IDF + SVM             | 0.90    |

**Insight:**  
Preprocessing and TF-IDF significantly improve model performance.

---

## Future Improvements  

- Implement deep learning models such as BERT  
- Extend to multilingual sentiment analysis  
- Improve sarcasm detection  
- Build real-time sentiment analysis system  

---

## Conclusion  

This project demonstrates how classical machine learning models can be effectively used for sentiment analysis. It highlights the importance of preprocessing and feature engineering, while also showing that SVM provides better performance compared to Naive Bayes in most scenarios.

---

## Acknowledgment  

- IMDB Dataset  
- Scikit-learn Library  
