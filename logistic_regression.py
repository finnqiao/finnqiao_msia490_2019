import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

df = pd.read_csv('preprocessed_text_with_bigrams.csv')

df.head()

scores = ['1','2','3','4','5']

# Assign x and y values
X = df['cleaned_text']
X_bi = df['cleaned_text_bigrams']
y = df['label']

# split into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

X_bi_train, X_bi_test, y_bi_train, y_bi_test = train_test_split(X_bi, y, test_size = 0.3, random_state = 42)

# base model results

logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression()),
               ])
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=scores))

# base model with uni_and_bigrams

logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression()),
               ])

logreg.fit(X_bi_train, y_bi_train)

y_bi_pred = logreg.predict(X_bi_test)

print('accuracy %s' % accuracy_score(y_bi_pred, y_bi_test))
print(classification_report(y_bi_test, y_bi_pred,target_names=scores))

# hyperparameter tuning, change solver to newton-cf to account for multi class
# also reduce C from default=1.0 to 0.5 to see impact of strengthening regularization

logreg2 = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(solver='newton-cg', C=0.5)),
               ])
logreg2.fit(X_train, y_train)

y_pred = logreg2.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=scores))

# repeat hyperparameter tuning on uni_and_bigrams

logreg2 = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(solver='newton-cg', C=0.5)),
               ])
logreg2.fit(X_train, y_train)

logreg2.fit(X_bi_train, y_bi_train)

y_bi_pred = logreg2.predict(X_bi_test)

print('accuracy %s' % accuracy_score(y_bi_pred, y_bi_test))
print(classification_report(y_bi_test, y_bi_pred,target_names=scores))
