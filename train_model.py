import pandas as pd
import re
import nltk
import pickle

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

nltk.download('stopwords')

# =============================
# Load Dataset (Kaggle)
# =============================
df = pd.read_csv("customer_support_tickets.csv")  # your file

# Rename for consistency
df = df.rename(columns={'Ticket Description': 'text'})

# =============================
# Text Cleaning
# =============================
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(clean_text)

# =============================
# Feature Extraction
# =============================
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])

# =============================
# CATEGORY MODEL
# =============================
y_category = df['Ticket Type']  # adjust if column name differs

X_train, X_test, y_train, y_test = train_test_split(X, y_category, test_size=0.2, random_state=42)

model_category = LogisticRegression(max_iter=200)
model_category.fit(X_train, y_train)

# =============================
# PRIORITY CREATION (if not present)
# =============================
def assign_priority(text):
    text = text.lower()
    if 'urgent' in text or 'error' in text or 'not working' in text:
        return 'High'
    elif 'slow' in text or 'delay' in text:
        return 'Medium'
    else:
        return 'Low'

if 'Priority' not in df.columns:
    df['Priority'] = df['text'].apply(assign_priority)

# =============================
# PRIORITY MODEL
# =============================
y_priority = df['Priority']

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X, y_priority, test_size=0.2, random_state=42)

model_priority = LogisticRegression(max_iter=200)
model_priority.fit(X_train_p, y_train_p)

# =============================
# SAVE FILES (IMPORTANT)
# =============================
pickle.dump(model_category, open('category_model.pkl', 'wb'))
pickle.dump(model_priority, open('priority_model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("✅ All .pkl files created successfully!")