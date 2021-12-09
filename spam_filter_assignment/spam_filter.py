# import required libraries
import matplotlib.pyplot as plt
import csv
import sklearn
import pickle
import pandas as pd
import numpy as np
import nltk

from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.model_selection import cross_val_score, learning_curve

import warnings
warnings.filterwarnings('ignore')

# stretch the output automatically
pd.options.display.width = 0

# loading the dataset
df = pd.read_csv(r"C:\Users\10inm\Desktop\ml_practice\Naive-Bayes_datasets\spam.csv",
                 encoding='latin-1')

# check the dataset
print(df.head())

# remove unwanted columns
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
# rename columns with a more descriptive name
df.rename(columns={'v2': 'message', 'v1': 'label'}, inplace=True)
# check everything worked
print(df.info())
# check for missing values
print(df.isna().sum())
# check how many legit and spam messages there is
print(df['label'].value_counts())

"""
Build a wordcloud to visually check word frequency.
"""

# create a corpus of spam messages
spam_words = ''
for val in df[df['label'] == 'spam'].text:
    text = val.lower()
    tokens = nltk.word_tokenize(text)
    for words in tokens:
        spam_words = spam_words + words + ' '

# create a corpus of legit messages
legit_words = ''
for val in df[df['label'] == 'spam'].text:
    text = val.lower()
    tokens = nltk.word_tokenize(text)
    for words in tokens:
        legit_words = legit_words + words + ' '

# generate the wordcloud for spam messages
spam_wordcloud = WordCloud(width=500, height=500).generate(spam_words)

plt.figure( figsize=(10,8), facecolor='w')
plt.imshow(spam_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

# generate the wordcloud for spam messages
legit_wordcloud = WordCloud(width=500, height=500).generate(legit_words)

plt.figure( figsize=(10,8), facecolor='w')
plt.imshow(legit_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()