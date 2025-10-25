# 📰 News Article Classification (Fake/Real)  

## 🎯 Objective
Classify news articles as **fake** or **real** using **Natural Language Processing (NLP)**.  
Users can input a news article, and the trained model predicts whether it is authentic or not.  

---

## ✨ Key Features
- 🧹 **Text Preprocessing** using NLTK (tokenization, stopwords removal, lemmatization)  
- 🔢 **TF-IDF Vectorization** for feature extraction  
- 🤖 **Model Training** with Logistic Regression and Naive Bayes  
- 📊 **Evaluation Metrics**: Accuracy, F1-score, Precision, Recall  
- 💻 **Interactive Demo** with Streamlit  
- 🗂️ **Modular Project Structure** for easy understanding  

---

## 🛠️ Tools & Technologies
- **Python 3.x**  
- **Pandas** – data handling  
- **Scikit-learn** – model training & evaluation  
- **NLTK** – text preprocessing  
- **Streamlit** – interactive demo  
- **Joblib** – saving and loading trained models  
- **Jupyter Notebook** – analysis and exploration  

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
