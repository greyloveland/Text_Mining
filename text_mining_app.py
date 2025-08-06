import streamlit as st
import pandas as pd
import joblib
import requests
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Setup for preprocessing
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words("english"))

def normalize_document(doc):
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc)
    doc = doc.lower().strip()
    tokens = word_tokenize(doc)
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

# Load data and models
df = pd.read_csv("df_file.csv")
model = joblib.load("best_text_classifier.pkl")
vectorizer = joblib.load("text_vectorizer.pkl")

# Gemini API setup
api_key = "AIzaSyCxjGu2TtUXDFZW1TjbqZe9qb75FxcNaRM"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

def ask_gemini(question, context):
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {"parts": [{"text": f"Context:\n{context}\n\nQuestion:\n{question}"}]}
        ]
    }
    response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return f"Error: {response.status_code} - {response.text}"

# Label mapping
label_names = {
    0: "Politics",
    1: "Sport",
    2: "Technology",
    3: "Entertainment",
    4: "Business"
}
# Human-readable cluster topics
cluster_terms = {
    0: ['game', 'england', 'said', 'win', 'cup', 'players', 'match', 'world', 'play', 'injury'],
    1: ['said', 'growth', 'economy', 'year', 'mr', 'bank', 'market', 'oil', 'sales', 'economic'],
    2: ['mr', 'election', 'labour', 'said', 'blair', 'party', 'government', 'brown', 'howard', 'prime'],
    3: ['film', 'best', 'said', 'awards', 'award', 'music', 'band', 'films', 'album', 'festival'],
    4: ['people', 'said', 'mobile', 'users', 'technology', 'music', 'digital', 'software', 'phone', 'broadband']
}


label_names_inv = {v: k for k, v in label_names.items()}

# Streamlit App
st.title("Lazy Reads")

# Dropdown for classification filter
selected_label_name = st.sidebar.selectbox("Filter by Classification Label", list(label_names.values()))
selected_label = label_names_inv[selected_label_name]

# Filter docs by selected label
filtered_df = df[df['Label'] == selected_label]

# Document selector
doc_index = st.sidebar.selectbox("Choose a document", filtered_df.index)
selected_doc = filtered_df.loc[doc_index]

# Display document
st.subheader("Document")
st.write(selected_doc['Text'])

# Display summary
st.subheader("Summary")
st.write(selected_doc['Summary'])

# Display known label (already stored)
st.subheader("Known Label")
st.write(selected_label_name)

# Show cluster info
st.subheader("Topic Cluster Terms")
cluster_id = selected_doc['Cluster']
top_words = ', '.join(cluster_terms.get(cluster_id, ["(unknown cluster)"]))
st.write(top_words)


# Ask a question
st.subheader("Ask a Question About This Document")
user_question = st.text_input("Your question:")
if st.button("Get Answer"):
    answer = ask_gemini(user_question, selected_doc['Text'])
    st.success(f"Answer: {answer}")
