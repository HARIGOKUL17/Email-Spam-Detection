import pandas as pd
df=pd.read_csv("C:\\Users\\booba\\Downloads\\spam.csv",encoding= 'latin')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import pickle

df['label'] = df['v1'].map({'ham': 0, 'spam': 1})
X = df['v2']
y = df['label']
# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)  # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
vect = cv.transform(data).toarray()

pickle.dump(clf, open("Spam_Model.pkl","wb"))