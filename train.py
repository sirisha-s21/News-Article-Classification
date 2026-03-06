
import pandas as pd
import joblib
import argparse
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from utils import clean_text

def load_data(path: Path):
    df = pd.read_csv(path)
    if 'text' in df.columns:
        df['content_clean'] = df['text'].fillna('').apply(clean_text)
    elif 'content' in df.columns:
        df['content_clean'] = df['content'].fillna('').apply(clean_text)
    elif 'title' in df.columns and 'text' in df.columns:
        df['content_clean'] = (df['title'].fillna('') + ' ' + df['text'].fillna('')).apply(clean_text)
    else:
        raise ValueError('No text column found.')
    if 'label' in df.columns:
        df['label_num'] = df['label'].apply(lambda x: 1 if str(x).strip().lower() in ('real','1','true','truth') else 0)
    else:
        raise ValueError('No label column found.')
    return df['content_clean'].values, df['label_num'].values

def build_and_eval(X_train, X_test, y_train, y_test):
    pipe_lr = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    pipe_nb = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000)),
        ('clf', MultinomialNB())
    ])

    print('Training Logistic Regression...')
    pipe_lr.fit(X_train, y_train)
    print('Training MultinomialNB...')
    pipe_nb.fit(X_train, y_train)

    for name, model in [('LogisticRegression', pipe_lr), ('MultinomialNB', pipe_nb)]:
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        print('\n===', name, '===')
        print('Accuracy:', acc)
        print('F1-score:', f1)
        print('Precision:', prec)
        print('Recall:', rec)
        print('Confusion matrix:\n', confusion_matrix(y_test, preds))
        print('\nClassification report:\n', classification_report(y_test, preds))

    joblib.dump(pipe_lr, 'model_lr.joblib')
    joblib.dump(pipe_nb, 'model_nb.joblib')
    print('\nSaved model_lr.joblib and model_nb.joblib')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/sample_news.csv')
    args = parser.parse_args()
    X, y = load_data(Path(args.data))
    if len(y) < 2:
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    build_and_eval(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()
