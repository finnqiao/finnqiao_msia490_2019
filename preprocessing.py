import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import gensim
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

df = pd.read_csv('/Users/finn/Downloads/amazon-fine-food-reviews/Reviews.csv')

# basic text preprocessing before passing documents into gensim

# lower case
df['cleaned_text'] = df.apply(lambda x: x['Text'].lower(), axis = 1)

# remove numbers
df['cleaned_text'] = df.apply(lambda x: re.sub(r'\d+', '', x['cleaned_text']), axis = 1)

# remove symbols and tokenize

tokenizer = RegexpTokenizer(r'\w+')

df['cleaned_text'] = df.apply(lambda x: tokenizer.tokenize(x['cleaned_text']), axis = 1)

# set stopwords

stop_words = set(stopwords.words('english'))

df['cleaned_text'] = df.apply(lambda x: [w for w in x['cleaned_text'] if w not in stop_words], axis=1)

# lemmatize words

lemmatizer=WordNetLemmatizer()

df['cleaned_text'] = df.apply(lambda x: [lemmatizer.lemmatize(w) for w in x['cleaned_text']], axis=1)

# lemmatizing didn't seem to change too many words, add stemming

stemmer= PorterStemmer()

df['cleaned_text'] = df.apply(lambda x: [stemmer.stem(w) for w in x['cleaned_text']], axis=1)


df.head()

df['label'] = df['Score'].astype('category', copy=False)

df[['cleaned_text','label']].to_csv('preprocessed_text.csv', index=False)

df['cleaned_text_bigrams'] = df.apply(lambda x: x['cleaned_text'] + list(map(lambda x: '_'.join(x), zip(x['cleaned_text'], x['cleaned_text'][1:]))), axis=1)

df[['cleaned_text', 'cleaned_text_bigrams','label']].to_csv('preprocessed_text_with_bigrams.csv', index=False)

mydict = gensim.corpora.Dictionary(df['cleaned_text'])

tfidf_model = gensim.models.TfidfModel(dictionary=mydict)

tfidf_model
