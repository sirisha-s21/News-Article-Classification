# 📰 News Article Classification (Fake / Real)

## 🎯 Objective
This project uses **Natural Language Processing (NLP)** to classify news articles as **Fake** or **Real**.

Users can enter a news article, and the trained machine learning model predicts whether the article is authentic or not.

---

## 🌐 Live Demo

🚀 **Try the deployed app:**  
👉 https://news-article-classification.streamlit.app/
---

## ✨ Key Features

- 🧹 **Text Preprocessing** using NLTK  
  - Tokenization  
  - Stopwords removal  
  - Lemmatization  

- 🔢 **TF-IDF Vectorization** for feature extraction  

- 🤖 **Machine Learning Models**
  - Logistic Regression
  - Naive Bayes

- 📊 **Evaluation Metrics**
  - Accuracy
  - Precision
  - Recall
  - F1-Score

- 💻 **Interactive Web Interface** built using Streamlit

- 🗂️ **Modular Project Structure** for easy understanding and reproducibility

---

## 🛠️ Tools & Technologies

- **Python 3.x**
- **Pandas** – data handling
- **Scikit-learn** – machine learning models
- **NLTK** – text preprocessing
- **Streamlit** – web interface
- **Joblib** – model serialization
- **Jupyter Notebook** – experimentation and analysis

---

## 🚀 How to Run Locally

1️⃣ Clone the repository
```bash
git clone https://github.com/sirisha-s21/News-Article-Classification.git
cd News_Article_Classifier
```
2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Merge the dataset
python merge_dataset.py
Creates data/news_dataset.csv ready for training.

4️⃣ Train the models
python train.py --data data/news_dataset.csv
Generates model_lr.joblib and model_nb.joblib.

5️⃣ Run the Streamlit app
streamlit run app.py
Open the local browser page and enter news articles to get predictions.
