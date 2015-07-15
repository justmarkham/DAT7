'''
CLASS: Naive Bayes SMS spam classifier
'''

## READING IN THE DATA

# read tab-separated file using pandas
import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/DAT7/master/data/sms.tsv'
sms = pd.read_table(url, sep='\t', header=None, names=['label', 'msg'])

# examine the data
sms.head(20)
sms.label.value_counts()
sms.msg.describe()

# convert label to a binary variable
sms['label'] = sms.label.map({'ham':0, 'spam':1})

# split into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(sms.msg, sms.label, random_state=1)
X_train.shape
X_test.shape


## COUNTVECTORIZER: 'convert text into a matrix of token counts'
## http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

from sklearn.feature_extraction.text import CountVectorizer

# start with a simple example
train_simple = ['call you tonight',
                'Call me a cab',
                'please call me... PLEASE!']

# learn the 'vocabulary' of the training data
vect = CountVectorizer()
vect.fit(train_simple)
vect.get_feature_names()

# transform training data into a 'document-term matrix'
train_simple_dtm = vect.transform(train_simple)
train_simple_dtm
train_simple_dtm.toarray()

# examine the vocabulary and document-term matrix together
pd.DataFrame(train_simple_dtm.toarray(), columns=vect.get_feature_names())

# transform testing data into a document-term matrix (using existing vocabulary)
test_simple = ["please don't call me"]
test_simple_dtm = vect.transform(test_simple)
test_simple_dtm.toarray()
pd.DataFrame(test_simple_dtm.toarray(), columns=vect.get_feature_names())


## USING COUNTVECTORIZER WITH SMS DATA

# instantiate the vectorizer
vect = CountVectorizer()

# learn vocabulary and create document-term matrix in a single step
train_dtm = vect.fit_transform(X_train)
train_dtm

# transform testing data into a document-term matrix
test_dtm = vect.transform(X_test)
test_dtm


## EXAMINING THE FEATURES AND THEIR COUNTS

# store feature names and examine them
train_features = vect.get_feature_names()
train_features[:50]
train_features[-50:]

# convert train_dtm to a regular array
train_arr = train_dtm.toarray()
train_arr

# count how many times EACH token appears across ALL messages in train_arr
import numpy as np
np.sum(train_arr, axis=0)

# create a DataFrame of tokens with their counts
train_token_counts = pd.DataFrame({'token':train_features, 'count':np.sum(train_arr, axis=0)})
train_token_counts.sort('count', ascending=False)


## MODEL BUILDING WITH NAIVE BAYES
## http://scikit-learn.org/stable/modules/naive_bayes.html

# train a Naive Bayes model using train_dtm
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(train_dtm, y_train)

# make predictions on test data using test_dtm
y_pred_class = nb.predict(test_dtm)

# compare predictions to true labels
from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred_class)
print metrics.confusion_matrix(y_test, y_pred_class)

# predict (poorly calibrated) probabilities and calculate AUC
y_pred_prob = nb.predict_proba(test_dtm)[:, 1]
y_pred_prob
print metrics.roc_auc_score(y_test, y_pred_prob)

# show the message text for the false positives
X_test[y_test < y_pred_class]

# show the message text for the false negatives
X_test[y_test > y_pred_class]


## COMPARING NAIVE BAYES WITH LOGISTIC REGRESSION

# instantiate/fit/predict
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)
logreg.fit(train_dtm, y_train)
y_pred_class = logreg.predict(test_dtm)
y_pred_prob = logreg.predict_proba(test_dtm)[:, 1]

# evaluate
print metrics.accuracy_score(y_test, y_pred_class)
print metrics.confusion_matrix(y_test, y_pred_class)
print metrics.roc_auc_score(y_test, y_pred_prob)

# show false positives and false negatives
X_test[y_test < y_pred_class]
X_test[y_test > y_pred_class]


## BONUS CONTENT: CALCULATING THE 'SPAMMINESS' OF EACH TOKEN

# create separate DataFrames for ham and spam
sms_ham = sms[sms.label==0]
sms_spam = sms[sms.label==1]

# learn the vocabulary of ALL messages and save it
vect.fit(sms.msg)
all_features = vect.get_feature_names()

# create document-term matrix of ham, then convert to a regular array
ham_dtm = vect.transform(sms_ham.msg)
ham_arr = ham_dtm.toarray()

# create document-term matrix of spam, then convert to a regular array
spam_dtm = vect.transform(sms_spam.msg)
spam_arr = spam_dtm.toarray()

# count how many times EACH token appears across ALL messages in ham_arr
ham_counts = np.sum(ham_arr, axis=0)

# count how many times EACH token appears across ALL messages in spam_arr
spam_counts = np.sum(spam_arr, axis=0)

# create a DataFrame of tokens with their separate ham and spam counts
all_token_counts = pd.DataFrame({'token':all_features, 'ham':ham_counts, 'spam':spam_counts})

# add one to ham counts and spam counts so that ratio calculations (below) make more sense
all_token_counts['ham'] = all_token_counts.ham + 1
all_token_counts['spam'] = all_token_counts.spam + 1

# calculate ratio of spam-to-ham for each token
all_token_counts['spam_ratio'] = all_token_counts.spam / all_token_counts.ham
all_token_counts.sort('spam_ratio')
