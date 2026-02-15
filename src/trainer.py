"""
sentiment model pickle saver
    - use logistic regression for 3-sentiment level classifying
    - use TF-IDF for text logic handling
    - output in JSON strings for Next.js POST request
    - save the model to `src/sentiment_model.pkl`

@goge052215
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    confusion_matrix,
    classification_report
)
from sklearn.pipeline import Pipeline
import preprocessing as pp

class Trainer:
    def __init__(self, train_df):
        self.raw_train_df = train_df
        self.preprocessor = pp.Preprocessor(train_df)
        self.preprocessor.preprocess()

    def _prepare_text_df(self, df):
        df = df[['text', 'sentiment']].copy()
        df['sentiment'] = df['sentiment'].map({
            'neutral': 0,
            'negative': 1,
            'positive': 2
        })
        df = df.dropna()
        return df

    def train_text(self, test_df):
        train_text_df = self._prepare_text_df(self.raw_train_df)
        test_text_df = self._prepare_text_df(test_df)

        X_train = train_text_df['text']
        y_train = train_text_df['sentiment']
        X_test = test_text_df['text']
        self.y_test = test_text_df['sentiment']

        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
            ("logreg", LogisticRegression(max_iter=1000))
        ])
        self.model.fit(X_train, y_train)
        self.y_pred = self.model.predict(X_test)

        result = {
            "accuracy": accuracy_score(self.y_test, self.y_pred),
            "precision": precision_score(self.y_test, self.y_pred, average='weighted'),
            "f1": f1_score(self.y_test, self.y_pred, average='weighted'),
            "recall": recall_score(self.y_test, self.y_pred, average='weighted'),
            "confusion_matrix": confusion_matrix(self.y_test, self.y_pred).tolist(),
            "classification_report": classification_report(self.y_test, self.y_pred)
        }

        return result

    def train(self, test_df, model_type="tfidf_logreg"):
        return self.train_text(test_df)
    
    # confusion matrix for testing and model dev
    def conf_matrix(self):
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            confusion_matrix(self.y_test, self.y_pred), 
            annot=True, fmt='d', cmap='Blues'
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def save_model(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
