import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import fasttext
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report



# Base models with learning rate 0.1 and 10 epochs
model_base = fasttext.train_supervised(input="fasttext.train", lr=0.1, epoch = 10)
model_bigram_base = fasttext.train_supervised(input="fasttext.train", wordNgrams=2, lr=0.1, epoch = 10)

# Further hyperparameter tuning on learning rate to increase learning rate but increasing to 25 epochs
model_2 = fasttext.train_supervised(input="fasttext.train", lr=1, epoch=25)
model_bigram_2 = fasttext.train_supervised(input="fasttext.train", wordNgrams=2, lr=1, epoch=25)

model_base.test('fasttext.valid')
model_bigram_base.test('fasttext.valid')
model_2.test('fasttext.valid')
model_bigram_2.test('fasttext.valid')
