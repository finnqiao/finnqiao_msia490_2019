import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/finn/Downloads/amazon-fine-food-reviews/Reviews.csv')

# Produce  a one- to two-sentence description of your dataset and a table with
# dataset summary statistics. The summary statistics should include at minimum the
# number of documents, number of labels, label distribution,
# average / mean word length of documents.

# Dimensions of dataframe
df.shape

# First 5 rows
df.head()

# Distribution of main 'score' label
df['Score'].value_counts()

# Descriptive stats on length of the text column
df['Text_Length'] = df.apply(lambda x: len(x['Text'].split(' ')),axis=1)

df['Text_Length'].mean()

df['Text_Length'].max()

df['Text_Length'].min()

df['Text_Length'].median()
