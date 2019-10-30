import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

df = pd.read_csv('preprocessed_text_with_bigrams.csv')

df.columns

# Assign x and y values
X = df['cleaned_text']
X_bi = df['cleaned_text_bigrams']
y = df['label']

# split into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

X_bi_train, X_bi_test, y_bi_train, y_bi_test = train_test_split(X_bi, y, test_size = 0.3, random_state = 42)

scores = ['1','2','3','4','5']

# base model results

svm = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LinearSVC()),
               ])
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=scores))

# base model with uni_and_bigrams

svm = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LinearSVC()),
               ])

svm.fit(X_bi_train, y_bi_train)

y_bi_pred = svm.predict(X_bi_test)

print('accuracy %s' % accuracy_score(y_bi_pred, y_bi_test))
print(classification_report(y_bi_test, y_bi_pred,target_names=scores))

# hyperparameter tuning, change loss function to squared_hinge to see impact of other loss function
# also reduce C from default=1.0 to 0.5 to see impact of strengthening regularization

svm2 = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LinearSVC(solver='squared_hinge', C=0.5)),
               ])
svm2.fit(X_train, y_train)

y_pred = svm2.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=scores))

# repeat hyperparameter tuning on uni_and_bigrams

svm2 = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LinearSVC(solver='squared_hinge', C=0.5)),
               ])
svm2.fit(X_train, y_train)

svm2.fit(X_bi_train, y_bi_train)

y_bi_pred = svm2.predict(X_bi_test)

print('accuracy %s' % accuracy_score(y_bi_pred, y_bi_test))
print(classification_report(y_bi_test, y_bi_pred,target_names=scores))
