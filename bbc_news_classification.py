import os
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import gensim.parsing.preprocessing as gsp
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('stopwords')
stopw = stopwords.words('english')
data_folder = "./dataset"
folders = ["business","entertainment","politics","sport","tech"]

# 1. Load raw data
os.chdir(data_folder)

x = []
y = []

for i in folders:
    files = os.listdir(i)
    for text_file in files:
        file_path = i + "/" +text_file
        with open(file_path, 'r', encoding='gbk', errors='ignore') as f:
            data = f.readlines()
        data = ' '.join(data)
        x.append(data)
        y.append(i)
   
data = {'text': x, 'category': y}       
df = pd.DataFrame(data)

df.to_csv('./bbc_news.csv', index=False)

# 2. Preprocess
df = pd.read_csv("bbc_news.csv")

# Count the number of words in each line of text, and plot the image
df['text_length'] = df['text'].apply(lambda x:len(x.split()))
plt.figure(figsize=(12,6),dpi = 300)
plt.hist(df['text_length'],bins = 40)
plt.xlabel("Word Number:")
plt.ylabel('Counts')
plt.title('Text Lengths')
plt.show()

# Plot the distribution of labels
print(df['category'].value_counts())
temp = df['category'].value_counts()
fig = plt.figure(figsize = (5,5),dpi =120)
sns.barplot(temp.index,temp.values)

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\']',' ',text)
    text = text.split()
    text = [word for word in text if word not in stopw]
    text = ' '.join(text)
    text = re.sub(r'  ', ' ', text)
    text = re.sub(r'   ', ' ', text)
    return text

df['clean_text'] = df['text'].apply(lambda x:clean_text(x))

# Feature extraction
# Word frequency feature
count_vectorizer = CountVectorizer()
X_word_freq = count_vectorizer.fit_transform(df['clean_text'])

# TF-IDF feature
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(df['clean_text'])

# Word embeddings feature
# Train Word2Vec model on the dataset
sentences = [text.split() for text in df['clean_text']]
word2vec_model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)

# Average Word2Vec embeddings
X_word2vec = np.array([np.mean([word2vec_model.wv[word] for word in sentence if word in word2vec_model.wv] or [np.zeros(100)], axis=0) for sentence in sentences])

# Combine features
X_combined = np.concatenate((X_tfidf.toarray(), X_word2vec), axis=1)

# Target variable
y = df['category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Macro-averaged Precision:", precision)
print("Macro-averaged Recall:", recall)
print("Macro-averaged F1:", f1)

# Plotting
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()