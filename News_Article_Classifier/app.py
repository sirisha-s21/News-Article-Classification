
import streamlit as st
import joblib
import numpy as np
from utils import clean_text

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="News Article Classifier",
    page_icon="📰",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#eef7f6,#f9ffff);
}

/* center container */
.block-container{
    max-width:750px;
    padding-top:2rem;
}

/* main title */
.main-title{
    text-align:center;
    font-size:48px;
    font-weight:800;
    color:#2c3e50;
    margin-bottom:5px;
}

/* subtitle */
.subtitle{
    text-align:center;
    font-size:18px;
    color:#5f6c7b;
    margin-bottom:40px;
}

/* card container */
.card{
    background:white;
    padding:30px;
    border-radius:14px;
    box-shadow:0px 6px 18px rgba(0,0,0,0.08);
}

/* prediction text */
.real{
    font-size:32px;
    font-weight:700;
    color:#27ae60;
    text-align:center;
}

.fake{
    font-size:32px;
    font-weight:700;
    color:#e74c3c;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">📰 Fake News Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI system to classify news articles as REAL or FAKE</div>', unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
try:
    model_lr = joblib.load("model_lr.joblib")
except:
    model_lr = None

try:
    model_nb = joblib.load("model_nb.joblib")
except:
    model_nb = None

# ---------------- INPUT CARD ----------------


st.markdown("### Choose Model")
model_choice = st.selectbox(
    "",
    ["LogisticRegression (LR)", "MultinomialNB (NB)"],
    label_visibility="collapsed"
)

title = st.text_input("Article Title")

text = st.text_area(
    "Paste News Article",
    height=220,
    placeholder="Paste the full news article here..."
)

predict = st.button("🔎 Analyze Article", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


# ---------------- LR EXPLANATION FUNCTION ----------------
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

        words = [(feature_names[i], float(contrib[i])) for i in top_idx if vals[i] > 0]

        return proba, idx, words

    except:
        return None, None, []


# ---------------- PREDICTION ----------------
if predict:

    if not text and not title:
        st.warning("⚠️ Enter article text or title.")

    else:

        combined = (title or "") + " " + (text or "")
        processed = clean_text(combined)

        st.write("")
        st.write("## 🧠 Prediction Result")

        # ---------- Logistic Regression ----------
        if model_choice.startswith("Logistic") and model_lr:

            proba, idx, words = explain_lr(model_lr, processed)

            label = "REAL" if idx == 1 else "FAKE"
            conf = proba.max()

            if label == "REAL":
                st.markdown('<p class="real">✅ REAL NEWS</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="fake">❌ FAKE NEWS</p>', unsafe_allow_html=True)

            st.progress(float(conf))
            st.write(f"Confidence: **{conf:.2f}**")

            if words:
                st.write("### 🔎 Important Words")
                for w, c in words:
                    st.write(f"• **{w}** (score {c:.4f})")

        # ---------- Naive Bayes ----------
        elif model_choice.startswith("Multinomial") and model_nb:

            pipe = model_nb
            tfidf = pipe.named_steps['tfidf']
            clf = pipe.named_steps['clf']

            v = tfidf.transform([processed])
            proba = clf.predict_proba(v)[0]

            idx = int(np.argmax(proba))
            label = "REAL" if idx == 1 else "FAKE"
            conf = proba.max()

            if label == "REAL":
                st.markdown('<p class="real">✅ REAL NEWS</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="fake">❌ FAKE NEWS</p>', unsafe_allow_html=True)

            st.progress(float(conf))
            st.write(f"Confidence: **{conf:.2f}**")

            try:
                feature_names = tfidf.get_feature_names_out()
                logp = clf.feature_log_prob_

                st.write("### 📊 Important Words")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("Top REAL words")
                    top_pos = logp[1].argsort()[-8:][::-1]
                    for i in top_pos:
                        st.write(feature_names[i])

                with col2:
                    st.write("Top FAKE words")
                    top_neg = logp[0].argsort()[-8:][::-1]
                    for i in top_neg:
                        st.write(feature_names[i])

            except:
                pass

        else:
            st.error("❌ Models not found. Run `python train.py` first.")

st.markdown("---")
