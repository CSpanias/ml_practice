# import required libraries
import matplotlib.pyplot as plt
import pandas as pd
import string
import nltk
import seaborn as sns

from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

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
for val in df[df['label'] == 'spam'].message:
    message = val.lower()
    tokens = nltk.word_tokenize(message)
    for words in tokens:
        spam_words = spam_words + words + ' '

# create a corpus of legit messages
legit_words = ''
for val in df[df['label'] == 'spam'].message:
    message = val.lower()
    tokens = nltk.word_tokenize(message)
    for words in tokens:
        legit_words = legit_words + words + ' '

# generate the wordcloud for spam messages
spam_wordcloud = WordCloud(width=500, height=500).generate(spam_words)

plt.figure(figsize=(10, 8), facecolor='w')
plt.imshow(spam_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

# generate the wordcloud for spam messages
legit_wordcloud = WordCloud(width=500, height=500).generate(legit_words)

plt.figure(figsize=(10, 8), facecolor='w')
plt.imshow(legit_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


def text_process(text_1):
    """Remove punctuation and stopwords."""
    text_1 = text_1.translate(str.maketrans('', '', string.punctuation))
    text_1 = [word for word in text_1.split() if word.lower() not in stopwords.words
    ('english')]

    return " ".join(text_1)


df['message'] = df['message'].apply(text_process)
print(df.head())

# create new dataframes from the processed data
message = pd.DataFrame(df['message'])
label = pd.DataFrame(df['label'])

# converting words to vectors
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(df['message'])
print(vectors.shape)

"""
Build ML models.
"""
# features = word vectors
features = vectors
# splitting data into train and test set
X_train, X_test, y_train, y_test = train_test_split(features, df['label'],
                                                    test_size=0.2, random_state=1)

# initialize multiple classification models
#initialize multiple classification models
svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier(n_neighbors=49)
mnb = MultinomialNB(alpha=0.2)
dtc = DecisionTreeClassifier(min_samples_split=7, random_state=111)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=31, random_state=111)

#create a dictionary of variables and models
clfs = {'SVC': svc, 'KN': knc, 'NB': mnb, 'DT': dtc, 'LR': lrc, 'RF': rfc}


#fit the data onto the models
def train(clf, features_1, targets):
    clf.fit(features_1, targets)


# make predictions
def predict(clf, features_2):
    return clf.predict(features_2)


# use a for loop to train and predict for all models
pred_scores_word_vectors = []
for k, v in clfs.items():
    train(v, X_train, y_train)
    pred = predict(v, X_test)
    pred_scores_word_vectors.append((k, [accuracy_score(y_test, pred)]))

# predictions
print(pred_scores_word_vectors)


# write functions to detect if the message is spam or not
def find(spam_or_not):
    if spam_or_not == 1:
        print("Message is SPAM")
    else:
        print("Message is NOT Spam")


new_text = ['You won 10000 dollars!']
integers = vectorizer.transform(new_text)
x = mnb.predict(integers)
find(x)

# Naive Bayes
y_pred_nb = mnb.predict(X_test)
y_true_nb = y_test
cm = confusion_matrix(y_true_nb, y_pred_nb)
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("y_pred_nb")
plt.ylabel("y_true_nb")
plt.show()