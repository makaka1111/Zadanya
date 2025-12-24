import re
import numpy as np
from collections import Counter
import math
import sys
FILENAME = '9.md'  # Фиксируем файл с ТЗ
with open(FILENAME, 'r', encoding='utf-8') as f:
text = f.read()
chunks = [c.strip() for c in re.split(r'(?m)^(?=#)', text) if c.strip()]
def preprocess(text):
text = text.lower()
text = re.sub(r'[^а-яa-z0-9\s]', '', text)  # Поддержка русского и английского
words = text.split()
return words
docs = [preprocess(chunk) for chunk in chunks]
vocab = set()
for doc in docs:
vocab.update(doc)
vocab = sorted(vocab)
word_to_id = {word: i for i, word in enumerate(vocab)}
V = len(vocab)
N = len(docs)
TF (term frequency)
tf = np.zeros((N, V))
for i, doc in enumerate(docs):
counts = Counter(doc)
doc_len = len(doc)
for word, count in counts.items():
tf[i, word_to_id[word]] = count / doc_len if doc_len > 0 else 0
DF и IDF
df = np.zeros(V)
for doc in docs:
for word in set(doc):
df[word_to_id[word]] += 1
idf = np.log(N / (df + 1))  # Сглаживание
TF-IDF
tfidf = tf * idf
norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
tfidf_norm = tfidf / (norms + 1e-8)
np.savez("index.npz", embs=tfidf_norm, texts=np.array(chunks, dtype=object), vocab=np.array(vocab, dtype=object), idf=idf)
print("Готово. Разделов:", len(chunks), "| Файл: index.npz | Словарь:", V)









