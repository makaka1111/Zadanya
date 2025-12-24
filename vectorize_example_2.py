import numpy as np
from collections import Counter
import re

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

text = "Мир"  # Пример из слайдов; замените на реальный запрос, например. "требования к интерфейсу"
doc = preprocess(text)
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

topk = sims.argsort()[-3:][::-1]
print("Top-3 совпадений:")
for rank, i in enumerate(topk, 1):
print(f"{rank}. idx={int(i)}  score={sims[i]:.4f}\n{texts[i]}\n---")
