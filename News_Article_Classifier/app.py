
import streamlit as st
import joblib
import numpy as np
from utils import clean_text

st.set_page_config(page_title='News Article Classifier', page_icon='📰')
st.title('News Article Classification — Fake / Real')

st.markdown('Paste an article and choose a model. Shows prediction, confidence, and explanation (top words).')

try:
    model_lr = joblib.load('model_lr.joblib')
except:
    model_lr = None
try:
    model_nb = joblib.load('model_nb.joblib')
except:
    model_nb = None

model_choice = st.selectbox('Choose model', ['LogisticRegression (LR)', 'MultinomialNB (NB)'])

title = st.text_input('Title (optional)')
text = st.text_area('Article text', height=250)

def explain_lr(model, processed):
    try:
        tfidf = model.named_steps['tfidf']
        clf = model.named_steps['clf']
        v = tfidf.transform([processed])
        proba = clf.predict_proba(v)[0]
        idx = int(np.argmax(proba))
        feature_names = tfidf.get_feature_names_out()
        coef = clf.coef_[0]
        vals = v.toarray()[0]
        contrib = vals * coef
        top_idx = contrib.argsort()[-8:][::-1]
        words = [(feature_names[i], float(contrib[i])) for i in top_idx if vals[i]>0]
        return proba, idx, words
    except:
        return None, None, []

if st.button('Predict'):
    if not text and not title:
        st.warning('Enter article text or title.')
    else:
        combined = (title or '') + ' ' + (text or '')
        processed = clean_text(combined)
        if model_choice.startswith('Logistic') and model_lr is not None:
            proba, idx, words = explain_lr(model_lr, processed)
            label = 'REAL' if idx==1 else 'FAKE'
            conf = proba.max() if proba is not None else None
            st.success(f'Prediction: {label} — confidence: {conf:.2f}' if conf is not None else f'Prediction: {label}')
            if words:
                st.markdown('**Top contributing words:**')
                for w,c in words:
                    st.write(f"{w} (contribution {c:.4f})")
        elif model_choice.startswith('Multinomial') and model_nb is not None:
            pipe = model_nb
            tfidf = pipe.named_steps['tfidf']
            clf = pipe.named_steps['clf']
            v = tfidf.transform([processed])
            proba = clf.predict_proba(v)[0]
            idx = int(np.argmax(proba))
            label = 'REAL' if idx==1 else 'FAKE'
            conf = proba.max()
            st.success(f'Prediction: {label} — confidence: {conf:.2f}')
            try:
                feature_names = tfidf.get_feature_names_out()
                logp = clf.feature_log_prob_
                top_pos = logp[1].argsort()[-8:][::-1]
                top_neg = logp[0].argsort()[-8:][::-1]
                st.markdown('**Top words for REAL class:**')
                for i in top_pos:
                    st.write(f"{feature_names[i]} (score {float(logp[1][i]):.4f})")
                st.markdown('**Top words for FAKE class:**')
                for i in top_neg:
                    st.write(f"{feature_names[i]} (score {float(logp[0][i]):.4f})")
            except:
                pass
        else:
            st.error('Models not found. Run python train.py to create model files.')

st.markdown('---')
