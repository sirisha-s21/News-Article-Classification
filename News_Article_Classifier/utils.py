
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def clean_text(s: str) -> str:
    if s is None:
        return ''
    s = str(s).lower()
    s = re.sub(r'http\S+',' ', s)
    s = re.sub(r'[^a-z0-9\s]',' ', s)
    tokens = nltk.word_tokenize(s)
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOPWORDS and len(t)>1]
    return ' '.join(tokens)
