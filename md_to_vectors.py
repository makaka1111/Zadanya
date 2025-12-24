import numpy as np
from collections import Counter
import re
import sys

def preprocess(text):
text = text.lower()
text = re.sub(r'[^а-яa-z0-9\s]', '', text)
words = text.split()
return words

z = np.load("index.npz", allow_pickle=True)
embs = z["embs"]
texts = z["texts"].tolist()
vocab = z["vocab"].tolist()
idf = z["idf"]
word_to_id = {word: i for i, word in enumerate(vocab)}
V = len(vocab)

if len(sys.argv) > 1:
query = sys.argv[1]
else:
query = input("Введите запрос: ")

doc = preprocess(query)
counts = Counter(doc)
tf = np.zeros(V)
doc_len = len(doc)
for word, count in counts.items():
if word in word_to_id:
tf[word_to_id[word]] = count / doc_len if doc_len > 0 else 0
tfidf = tf * idf
norm = np.linalg.norm(tfidf)
emb = tfidf / (norm + 1e-8) if norm > 0 else tfidf

sims = embs @ emb
Топ-3 чанков
topk = sims.argsort()[-3:][::-1]
context = "\n\n".join([texts[i] for i in topk])

prompt = f"Используя следующий контекст из ТЗ на игру 'Властелин Земель':\n\n{context}\n\nОтветь на запрос: {query}"
print("Готовый промпт для отправки в нейронку (LLM):\n")
print(prompt)

import requests
response = requests.post("https://api.example.com/chat", json={"prompt": prompt, "model": "gpt-4"})
print(response.json()["answer"])
