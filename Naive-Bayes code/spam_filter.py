"""
Assignment for the AI & ML Bootcamp organized by UoH and IoC.
More info:
https://instituteofcoding.org/skillsbootcamps/course/skills-bootcamp-in-artificial-intelligence/

Guided tutorial on how to create a spam filter using nltk library.
More info:
https://github.com/tejank10/Spam-or-Ham/blob/master/spam_ham.ipynb
"""
# import required libraries
import pandas as pd # loading data
import numpy as np # generating random probabilities
import matplotlib.pyplot as plt # visualization
from nltk.tokenize import word_tokenize # messaging processing
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from math import log, sqrt # math functions
from wordcloud import WordCloud # visualization

# automatically stretch output relative to terminal size
pd.options.display.width = 0

# loading data
df = pd.read_csv(r"C:\Users\10inm\Desktop\ml_practice\Naive-Bayes_datasets\spam.csv",
                 encoding='latin-1')

# check data (data types, rows x cols, null-values)
print(df.info())
# check first five rows
print(df.head())
# check for missing values
print(df.isna().sum())
# delete unnecessary columns
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
# rename columns with a more descriptive word
df.rename({'v1': 'labels', 'v2': 'message'}, axis=1, inplace=True)
# check target size
print(df['labels'].value_counts())
# convert categorical variable ('label') to binary (0/1) using one-hot encoding
#df = pd.get_dummies(df, columns=['label'])
# assign 0 for ham and 1 for spam
df['label'] = df['labels'].map({'ham': 0, 'spam': 1})
# drop labels columns
df.drop(['labels'], axis=1, inplace=True)
# check everything is done correctly
print(df.head())

total_mails = df['message'].shape[0]
train_index, test_index = list(), list()
for i in range(df.shape[0]):
    if np.random.uniform(0, 1) < 0.75:
        train_index += [i]
    else:
        test_index += [i]
train_data = df.loc[train_index]
test_data = df.loc[test_index]

train_data.reset_index(inplace=True)
train_data.drop(['index'], axis=1, inplace=True)
print(train_data.head())

# generate a word cloud
spam_words = ' '.join(list(df[df['label'] == 1]['message']))
spam_wc = WordCloud(width=512, height=512).generate(spam_words)
plt.figure(figsize=(10, 8), facecolor='k')
plt.imshow(spam_wc)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

def process_message(message, lower_case=True, stem=True, stop_words=True,
                    gram=2):

    if lower_case:
        # all characters lowercase
        message = message.lower()
    # tokenization (splitting up a message into pieces)
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w
    # remove stopwords
    if stop_words:
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]
    # stemming (keep the root of the word)
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    return words

# Bag of Words (term frequency)
# TF-IDF (Term Frequency-Inverse Document Frequency)
# Additive Smoothing (avoid encountering new word and set P(w)=0
# Laplace Smoothing (alpha=1)
class SpamClassifier(object):
    """"""
    def __init__(self, train_data, method='tf-idf'):
        """"""
        self.df, self.labels = train_data['message'], train_data['label']
        self.method = method

    def train(self):
        """"""
        self.calc_tf_and_idf()
        if self.method == 'tf-idf':
            self.calc_tf_idf()
        else:
            self.calc_prob()

    def calc_prob(self):
        """"""
        self.prob_spam = dict()
        self.prob_ham = dict()
        for word in self.tf_spam:
            self.prob_spam[word] = (self.tf_spam[word] + 1) / (self.spam_words + \
                                                               len(list(self.tf_spam.keys())))
        for word in self.tf_ham:
            self.prob_ham[word] = (self.tf_ham[word] + 1) / (self.ham_words + \
                                                             len(list(self.tf_ham.keys())))
        self.prob_spam_mail, self.prob_ham_mail = self.spam_mails / self.total_mails, self.ham_mails / self.total_mails

    def calc_tf_and_idf(self):
        """"""
        no_of_messages = self.df.shape[0]
        self.spam_mails, self.ham_mails = self.labels.value_counts()[1], self.labels.value_counts()[0]
        self.total_mails = self.spam_mails + self.ham_mails
        self.spam_words = 0
        self.ham_words = 0
        self.tf_spam = dict()
        self.tf_ham = dict()
        self.idf_spam = dict()
        self.idf_ham = dict()
        for i in range(no_of_messages):
            message_processed = process_message(self.df[i])
            # To keep track of whether the word has occurred in the message or not.
            count = list()
            # For IDF
            for word in message_processed:
                if self.labels[i]:
                    self.tf_spam[word] = self.tf_spam.get(word, 0) + 1
                    self.spam_words += 1
                else:
                    self.tf_ham[word] = self.tf_ham.get(word, 0) + 1
                    self.ham_words += 1
                if word not in count:
                    count += [word]
            for word in count:
                if self.labels[i]:
                    self.idf_spam[word] = self.idf_spam.get(word, 0) + 1
                else:
                    self.idf_ham[word] = self.idf_ham.get(word, 0) + 1

    def calc_tf_idf(self):
        """"""
        self.prob_spam = dict()
        self.prob_ham = dict()
        self.sum_tf_idf_spam = 0
        self.sum_tf_idf_ham = 0
        for word in self.tf_spam:
            self.prob_spam[word] = (self.tf_spam[word])\
                                   * log((self.spam_mails + self.ham_mails) \
                                    / (self.idf_spam[word] +
                                       self.idf_ham.get(word, 0)))
            self.sum_tf_idf_spam += self.prob_spam[word]
        for word in self.tf_spam:
            self.prob_spam[word] = (self.prob_spam[word] + 1) / (
                        self.sum_tf_idf_spam + len(list(self.prob_spam.keys())))

        for word in self.tf_ham:
            self.prob_ham[word] = (self.tf_ham[word]) * \
                                  log((self.spam_mails + self.ham_mails) \
                                / (self.idf_spam.get(word, 0) + self.idf_ham[word]))
            self.sum_tf_idf_ham += self.prob_ham[word]
        for word in self.tf_ham:
            self.prob_ham[word] = (self.prob_ham[word] + 1) /\
                                  (self.sum_tf_idf_ham +
                                   len(list(self.prob_ham.keys())))

        self.prob_spam_mail, self.prob_ham_mail = self.spam_mails / self.total_mails, self.ham_mails / self.total_mails

    def classify(self, processed_message):
        """"""
        pSpam, pHam = 0, 0
        for word in processed_message:
            if word in self.prob_spam:
                pSpam += log(self.prob_spam[word])
            else:
                if self.method == 'tf-idf':
                    pSpam -= log(self.sum_tf_idf_spam + len(list(self.prob_spam.keys())))
                else:
                    pSpam -= log(self.spam_words + len(list(self.prob_spam.keys())))
            if word in self.prob_ham:
                pHam += log(self.prob_ham[word])
            else:
                if self.method == 'tf-idf':
                    pHam -= log(self.sum_tf_idf_ham + len(list(self.prob_ham.keys())))
                else:
                    pHam -= log(self.ham_words + len(list(self.prob_ham.keys())))
            pSpam += log(self.prob_spam_mail)
            pHam += log(self.prob_ham_mail)
        return pSpam >= pHam

    def predict(self, test_data):
        """"""

        result = dict()
        for (i, message) in enumerate(test_data):
            processed_message = process_message(message)
            result[i] = int(self.classify(processed_message))
        return result


def metrics(labels, predictions):
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for z in range(len(labels)):
        true_pos += int(labels[z] == 1 and predictions[z] == 1)
        true_neg += int(labels[z] == 0 and predictions[z] == 0)
        false_pos += int(labels[z] == 0 and predictions[z] == 1)
        false_neg += int(labels[z] == 1 and predictions[z] == 0)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f_score = 2 * precision * recall / (precision + recall)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F-score: ", f_score)
    print("Accuracy: ", accuracy)

sc_tf_idf = SpamClassifier(train_data, 'tf-idf')
sc_tf_idf.train()
pred_tf_idf = sc_tf_idf.predict(test_data['message'])
print(metrics(test_data['label'], pred_tf_idf))

sc_bow = SpamClassifier(train_data, 'bow')
sc_bow.train()
preds_bow = sc_bow.predict(test_data['message'])
print(metrics(test_data['label'], preds_bow))

# test new messages
pm = process_message('I cant pick the phone right now. Pls send a message')
print(sc_tf_idf.classify(pm))

pm = process_message('Congratulations ur awarded $500 ')
print(sc_tf_idf.classify(pm))