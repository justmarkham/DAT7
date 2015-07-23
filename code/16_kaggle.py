'''
CLASS: Kaggle Stack Overflow competition
'''

# read in the file and set the first column as the index
import pandas as pd
train = pd.read_csv('train.csv', index_col=0)
train.head()


'''
What are some assumptions and theories to test?

OwnerUserId: not unique within the dataset, assigned in order
OwnerCreationDate: users with older accounts have more open questions
ReputationAtPostCreation: higher reputation users have more open questions
OwnerUndeletedAnswerCountAtPostTime: users with more answers have more open questions
Title and BodyMarkdown: well-written questions are more likely to be open
Tags: 1 to 5 tags are required, many unique tags
OpenStatus: Most questions should be open (encoded as 1)
'''

## OPEN STATUS

# dataset is perfectly balanced in terms of OpenStatus (not a representative sample)
train.OpenStatus.value_counts()


## USER ID

# OwnerUserId is not unique within the dataset, let's examine the top 3 users
train.OwnerUserId.value_counts()

# mostly closed questions, few answers, all lowercase, grammatical mistakes
train[train.OwnerUserId==466534].describe()
train[train.OwnerUserId==466534].head()

# fewer closed questions, high reputation but few answers, better grammar
train[train.OwnerUserId==39677].describe()
train[train.OwnerUserId==39677].head()

# same proportion of closed questions, lots of answers
train[train.OwnerUserId==34537].describe()
train[train.OwnerUserId==34537].head()


## REPUTATION

# ReputationAtPostCreation is higher for open questions: possibly use as a feature
train.groupby('OpenStatus').ReputationAtPostCreation.describe()

# not a useful histogram
train.ReputationAtPostCreation.hist()

# much more useful histogram
train[train.ReputationAtPostCreation < 1000].ReputationAtPostCreation.hist()

# grouped histogram
train[train.ReputationAtPostCreation < 1000].ReputationAtPostCreation.hist(by=train.OpenStatus, sharey=True)


## ANSWER COUNT

# rename column
train.rename(columns={'OwnerUndeletedAnswerCountAtPostTime':'Answers'}, inplace=True)

# Answers is higher for open questions: possibly use as a feature
train.groupby('OpenStatus').Answers.describe()

# grouped histogram
train[train.Answers < 50].Answers.hist(by=train.OpenStatus, sharey=True)


'''
Define a function that takes a raw CSV file and returns a DataFrame that
includes all created features (and any other modifications)
'''

# define the function
def make_features(filename):
    df = pd.read_csv(filename, index_col=0)
    df.rename(columns={'OwnerUndeletedAnswerCountAtPostTime':'Answers'}, inplace=True)
    return df

# apply function to both training and testing files
train = make_features('train.csv')
test = make_features('test.csv')


'''
Evaluate a model with two features
'''

# define X and y
feature_cols = ['ReputationAtPostCreation', 'Answers']
X = train[feature_cols]
y = train.OpenStatus

# split into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# fit a logistic regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)
logreg.fit(X_train, y_train)

# examine the coefficients to check that they makes sense
logreg.coef_

# predict response classes and predict class probabilities
y_pred_class = logreg.predict(X_test)
y_pred_prob = logreg.predict_proba(X_test)[:, 1]

# check how well we did
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)    # 0.543 (better than guessing)
metrics.confusion_matrix(y_test, y_pred_class)  # predicts closed most of the time
metrics.roc_auc_score(y_test, y_pred_prob)      # 0.607 (not horrible)
metrics.log_loss(y_test, y_pred_prob)           # 0.690 (what is this?)

# let's see if cross-validation gives us similar results
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(logreg, X, y, scoring='log_loss', cv=10)
scores.mean()       # 0.690 (identical to train/test split)
scores.std()        # very small


'''
Understanding log loss
'''

# 5 pretend response values
y_test = [0, 0, 0, 1, 1]

# 5 sets of predicted probabilities for those observations
y_pred_prob_sets = [[0.1, 0.2, 0.3, 0.8, 0.9],
                    [0.4, 0.4, 0.4, 0.6, 0.6],
                    [0.4, 0.4, 0.7, 0.6, 0.6],
                    [0.4, 0.4, 0.9, 0.6, 0.6],
                    [0.5, 0.5, 0.5, 0.5, 0.5]]

# calculate AUC for each set of predicted probabilities
for y_pred_prob in y_pred_prob_sets:
    print y_pred_prob, metrics.roc_auc_score(y_test, y_pred_prob)

# calculate log loss for each set of predicted probabilities
for y_pred_prob in y_pred_prob_sets:
    print y_pred_prob, metrics.log_loss(y_test, y_pred_prob)


'''
Create a submission file
'''

# train the model on ALL data (not X_train and y_train)
logreg.fit(X, y)

# predict class probabilities for the actual testing data (not X_test)
X_oos = test[feature_cols]
oos_pred_prob = logreg.predict_proba(X_oos)[:, 1]

# sample submission file indicates we need two columns: PostId and predicted probability
test.index      # PostId
oos_pred_prob   # predicted probability

# create a DataFrame that has 'id' as the index, then export to a CSV file
sub = pd.DataFrame({'id':test.index, 'OpenStatus':oos_pred_prob}).set_index('id')
sub.to_csv('sub1.csv')  # 0.694


'''
Explore data and create more features
'''

## TITLE

# create a new feature that represents the length of the title (in characters)
train['TitleLength'] = train.Title.apply(len)

# Title is longer for open questions: possibly use as a feature
train.TitleLength.hist(by=train.OpenStatus)


## BODY

# create a new feature that represents the length of the body (in characters)
train['BodyLength'] = train.BodyMarkdown.apply(len)

# BodyMarkdown is longer for open questions: possibly use as a feature
train[train.BodyLength < 5000].BodyLength.hist(by=train.OpenStatus)


## TAGS

# Tag1 is required, and the rest are optional
train.isnull().sum()

# there are over 5000 unique tags
train.Tag1.nunique()

# calculate the percentage of open questions for each tag
train.groupby('Tag1').OpenStatus.mean()

# percentage of open questions varies widely by tag (among popular tags)
train.groupby('Tag1').OpenStatus.agg(['mean','count']).sort('count')

# create a new feature that represents the number of tags for each question
train['NumTags'] = train.loc[:, 'Tag1':'Tag5'].notnull().sum(axis=1)

# NumTags is higher for open questions: possibly use as a feature
train.NumTags.hist(by=train.OpenStatus)


## USER ID

# OwnerUserId is assigned in numerical order
train.sort('OwnerUserId').OwnerCreationDate

# OwnerUserId is lower for open questions: possibly use as a feature
train.groupby('OpenStatus').OwnerUserId.describe()

# create a new feature that represents account age at time of question
train['OwnerCreationDate'] = pd.to_datetime(train.OwnerCreationDate)
train['PostCreationDate'] = pd.to_datetime(train.PostCreationDate)
train['OwnerAge'] = (train.PostCreationDate - train.OwnerCreationDate).dt.days


'''
Compare feature sets using cross-validation
'''

feature_cols = ['ReputationAtPostCreation', 'Answers', 'TitleLength', 'BodyLength', 'NumTags']
X = train[feature_cols]
cross_val_score(logreg, X, y, scoring='log_loss', cv=10).mean()     # 0.677

feature_cols = ['ReputationAtPostCreation', 'Answers', 'TitleLength', 'BodyLength', 'NumTags', 'OwnerUserId']
X = train[feature_cols]
cross_val_score(logreg, X, y, scoring='log_loss', cv=10).mean()     # 0.665

feature_cols = ['ReputationAtPostCreation', 'Answers', 'TitleLength', 'BodyLength', 'NumTags', 'OwnerAge']
X = train[feature_cols]
cross_val_score(logreg, X, y, scoring='log_loss', cv=10).mean()     # 0.672


'''
Update make_features and create another submission file
'''

# update the function
def make_features(filename):
    df = pd.read_csv(filename, index_col=0)
    df.rename(columns={'OwnerUndeletedAnswerCountAtPostTime':'Answers'}, inplace=True)
    df['TitleLength'] = df.Title.apply(len)
    df['BodyLength'] = df.BodyMarkdown.apply(len)
    df['NumTags'] = df.loc[:, 'Tag1':'Tag5'].notnull().sum(axis=1)
    df['OwnerCreationDate'] = pd.to_datetime(df.OwnerCreationDate)
    df['PostCreationDate'] = pd.to_datetime(df.PostCreationDate)
    df['OwnerAge'] = (df.PostCreationDate - df.OwnerCreationDate).dt.days
    return df

# apply function to both training and testing files
train = make_features('train.csv')
test = make_features('test.csv')

# train the model with OwnerUserId
feature_cols = ['ReputationAtPostCreation', 'Answers', 'TitleLength', 'BodyLength', 'NumTags', 'OwnerUserId']
X = train[feature_cols]
logreg.fit(X, y)

# predict class probabilities for the actual testing data
X_oos = test[feature_cols]
oos_pred_prob = logreg.predict_proba(X_oos)[:, 1]
oos_pred_prob

# create submission file
sub = pd.DataFrame({'id':test.index, 'OpenStatus':oos_pred_prob}).set_index('id')
sub.to_csv('sub2.csv')  # 0.864

# repeat with OwnerAge
feature_cols = ['ReputationAtPostCreation', 'Answers', 'TitleLength', 'BodyLength', 'NumTags', 'OwnerAge']
X = train[feature_cols]
logreg.fit(X, y)
X_oos = test[feature_cols]
oos_pred_prob = logreg.predict_proba(X_oos)[:, 1]
sub = pd.DataFrame({'id':test.index, 'OpenStatus':oos_pred_prob}).set_index('id')
sub.to_csv('sub3.csv')  # 0.637

# OwnerUserId overfits the training data, whereas OwnerAge does not
train.PostCreationDate.describe()
test.PostCreationDate.describe()


'''
Build a document-term matrix from Title using CountVectorizer
'''

# use CountVectorizer with the default settings
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
dtm = vect.fit_transform(train.Title)

# slightly improper cross-validation of a Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
cross_val_score(nb, dtm, train.OpenStatus, scoring='log_loss', cv=10).mean()    # 0.657

# try tuning CountVectorizer and repeat Naive Bayes
vect = CountVectorizer(stop_words='english')
dtm = vect.fit_transform(train.Title)
cross_val_score(nb, dtm, train.OpenStatus, scoring='log_loss', cv=10).mean()    # 0.635

# build document-term matrix for the actual testing data and make predictions
oos_dtm = vect.transform(test.Title)
nb.fit(dtm, train.OpenStatus)
oos_pred_prob = nb.predict_proba(oos_dtm)[:, 1]
sub = pd.DataFrame({'id':test.index, 'OpenStatus':oos_pred_prob}).set_index('id')
sub.to_csv('sub4.csv')  # 0.543


'''
BONUS: Dummy encoding of Tag1
'''

# number of unique tags for Tag1
train.Tag1.nunique()

# convert Tag1 from strings to integers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['Tag1_enc'] = le.fit_transform(train.Tag1)

# confirm that the conversion worked
train.Tag1.value_counts().head()
train.Tag1_enc.value_counts().head()

# create a dummy column for each value of Tag1_enc (returns a sparse matrix)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
tag1_dummies = ohe.fit_transform(train[['Tag1_enc']])
tag1_dummies

# try a Naive Bayes model with tag1_dummies as the features
cross_val_score(nb, tag1_dummies, train.OpenStatus, scoring='log_loss', cv=10).mean()   # 0.650

# adjust Tag1 on testing set since LabelEncoder errors on new values during a transform
test['Tag1'] = test['Tag1'].map(lambda s: '<unknown>' if s not in le.classes_ else s)
import numpy as np
le.classes_ = np.append(le.classes_, '<unknown>')

# apply the same encoding to the actual testing data and make predictions
test['Tag1_enc'] = le.transform(test.Tag1)
oos_tag1_dummies = ohe.transform(test[['Tag1_enc']])
nb.fit(tag1_dummies, train.OpenStatus)
oos_pred_prob = nb.predict_proba(oos_tag1_dummies)[:, 1]
sub = pd.DataFrame({'id':test.index, 'OpenStatus':oos_pred_prob}).set_index('id')
sub.to_csv('sub5.csv')  # 0.649
