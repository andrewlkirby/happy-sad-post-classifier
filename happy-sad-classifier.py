import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import numpy as np
from sklearn.utils import shuffle

seed = 3


df = pd.read_csv(r"data.csv")
df = df.dropna()
print(df.head())
print(df.iloc[0])
print("length: ", len(df))

#Sentiment + Subjectivty scores; LR

X = df[['compound', 'subj_scores']]
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed)

clf = LogisticRegression(penalty="l1", C=1.0, solver="liblinear").fit(
    X_train, y_train)

print("\n")
print("score with subjectivity + sentiment: ", clf.score(X_test, y_test))


#TFIDF + LR
X = df['body']
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed)

count_vectorizer = CountVectorizer(ngram_range = (1, 1))
tfidf = TfidfTransformer()

count_matrix_train = count_vectorizer.fit_transform(X_train)
X_train_tfidf = tfidf.fit_transform(count_matrix_train)
X_train_tfidf = X_train_tfidf.todense()

count_matrix_test = count_vectorizer.transform(X_test)
X_test_tfidf = tfidf.transform(count_matrix_test)
X_test_tfidf = X_test_tfidf.todense()

clf2 = LogisticRegression(penalty="l1", C=1.0, solver="liblinear").fit(
    X_train_tfidf, y_train)

print("\n")
print("score with TF-IDF: ", clf2.score(X_test_tfidf, y_test))


"""
#spacy vector maker & LR
nlp = spacy.load('en_core_web_sm')

df = shuffle(df, random_state=seed)
df_spacy = df.iloc[:1000]
X = df_spacy['body']
y = df_spacy['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed)

X_train_tok = X_train.apply(nlp)
X_train_vec = X_train_tok.apply(lambda sent: np.mean([token.vector for token in 
                                                  sent if not token.is_stop]))
X_train_vec = X_train_vec.values.reshape(-1, 1)

X_test_tok = X_test.apply(nlp)
X_test_vec = X_test_tok.apply(lambda sent: np.mean([token.vector for token in 
                                                  sent if not token.is_stop]))
#gives mean of all vectors; excludes stop words via if not token.is_stop                                        
#https://stackoverflow.com/questions/62676136/extract-sentence-embeddings-features-with-pandas-and-spacy
X_test_vec = X_test_vec.values.reshape(-1, 1)

clf3 = LogisticRegression(penalty="l1", C=1.0, solver="liblinear").fit(
    X_train_vec, y_train)

print("\n")
print("score with spacy vec: ", clf3.score(X_test_vec, y_test))
"""