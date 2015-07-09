# # Advanced Model Evaluation

# ## Agenda
# 1. Null accuracy, handling missing values
# 2. Confusion matrix, sensitivity, specificity, setting a threshold
# 3. Handling categorical features, interpreting logistic regression coefficients
# 4. ROC curves, AUC


# ## Part 1: Null Accuracy, Handling Missing Values

# ### Recap of the Titanic exercise

# TASK 1: read the data from titanic.csv into a DataFrame
import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/DAT7/master/data/titanic.csv'
titanic = pd.read_csv(url, index_col='PassengerId')

# TASK 2: define Pclass/Parch as the features and Survived as the response
feature_cols = ['Pclass', 'Parch']
X = titanic[feature_cols]
y = titanic.Survived

# TASK 3: split the data into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# TASK 4: fit a logistic regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)
logreg.fit(X_train, y_train)

# TASK 5: make predictions on testing set and calculate accuracy
y_pred_class = logreg.predict(X_test)
from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred_class)

# ### Null accuracy
# Null accuracy is the accuracy that could be achieved by always predicting the **most frequent class**. It is a baseline against which you may want to measure your classifier.

# compute null accuracy manually
print y_test.mean()
print 1 - y_test.mean()

# equivalent function in scikit-learn
from sklearn.dummy import DummyClassifier
dumb = DummyClassifier(strategy='most_frequent')
dumb.fit(X_train, y_train)
y_dumb_class = dumb.predict(X_test)
print metrics.accuracy_score(y_test, y_dumb_class)

# ### Handling missing values
# scikit-learn models expect that all values are **numeric** and **hold meaning**. Thus, missing values are not allowed by scikit-learn.

# One possible strategy is to just **drop missing values**:

# check for missing values
titanic.isnull().sum()

# drop rows with any missing values
titanic.dropna().shape

# drop rows where Age is missing
titanic[titanic.Age.notnull()].shape

# Sometimes a better strategy is to **impute missing values**:

# fill missing values for Age with the mean age
titanic.Age.fillna(titanic.Age.mean(), inplace=True)

# equivalent function in scikit-learn, supports mean/median/most_frequent
from sklearn.preprocessing import Imputer
imp = Imputer(strategy='mean', axis=1)
titanic['Age'] = imp.fit_transform(titanic.Age).T

# include Age as a feature
feature_cols = ['Pclass', 'Parch', 'Age']
X = titanic[feature_cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
logreg.fit(X_train, y_train)
y_pred_class = logreg.predict(X_test)
print metrics.accuracy_score(y_test, y_pred_class)


# ## Part 2: Confusion Matrix

# confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)

# calculate the sensitivity
43 / float(52 + 43)

# calculate the specificity
107 / float(107 + 21)

# store the predicted probabilities
y_pred_prob = logreg.predict_proba(X_test)[:, 1]

# plot the predicted probabilities
import matplotlib.pyplot as plt
plt.hist(y_pred_prob)
plt.xlabel('Predicted probability of survival')
plt.ylabel('Frequency')

# change the threshold for predicting survived to increase sensitivity
import numpy as np
y_pred_class = np.where(y_pred_prob > 0.25, 1, 0)

# equivalent function in scikit-learn
from sklearn.preprocessing import binarize
y_pred_class = binarize(y_pred_prob, 0.25)

# new confusion matrix
print metrics.confusion_matrix(y_test, y_pred_class)

# new sensitivity
print 68 / float(27 + 68)

# new specificity
print 57 / float(57 + 71)


# ## Part 3: Handling Categorical Features

# scikit-learn expects all features to be numeric. So how do we include a categorical feature in our model?
# - **Ordered categories:** transform them to sensible numeric values (example: small=1, medium=2, large=3)
# - **Unordered categories:** use dummy encoding

# **Pclass** is an ordered categorical feature, and is already encoded as 1/2/3, so we leave it as-is.
# **Sex** is an unordered categorical feature, and needs to be dummy encoded.

# ### Dummy encoding with two levels

# encode Sex_Female feature
titanic['Sex_Female'] = titanic.Sex.map({'male':0, 'female':1})

# include Sex_Female in the model
feature_cols = ['Pclass', 'Parch', 'Age', 'Sex_Female']
X = titanic[feature_cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
logreg=LogisticRegression(C=1e9)
logreg.fit(X_train, y_train)

# ### Logistic regression coefficients

zip(feature_cols, logreg.coef_[0])

# $$\log \left({p\over 1-p}\right) = \beta_0 + \beta_1x_1 + \beta_2x_2 + \beta_3x_3 + \beta_4x_4$$

# convert log-odds to odds
zip(feature_cols, np.exp(logreg.coef_[0]))

# Predict probability of survival for **Adam**: first class, no parents or kids, 29 years old, male.
logreg.predict_proba([1, 0, 29, 0])[:, 1]

# ### Interpreting the Pclass coefficient

# Predict probability of survival for **Bill**: same as Adam, except second class.
logreg.predict_proba([2, 0, 29, 0])[:, 1]

# How could we have calculated that change ourselves using the coefficients?

# $$odds = \frac {probability} {1 - probability}$$
# $$probability = \frac {odds} {1 + odds}$$

# convert Adam's probability to odds
adamodds = 0.5/(1 - 0.5)

# adjust odds for Bill due to lower class
billodds = adamodds * 0.295

# convert Bill's odds to probability
billodds/(1 + billodds)

# ### Interpreting the Sex_Female coefficient

# Predict probability of survival for **Susan**: same as Adam, except female.
logreg.predict_proba([1, 0, 29, 1])[:, 1]

# Let's calculate that change ourselves:

# adjust odds for Susan due to her sex
susanodds = adamodds * 14.6

# convert Susan's odds to probability
susanodds/(1 + susanodds)

# How do we interpret the **Sex_Female coefficient**? For a given Pclass/Parch/Age, being female is associated with an increase in the **log-odds of survival** by 2.68 (or an increase in the **odds of survival** by 14.6) as compared to a male, which is called the **baseline level**.

# What if we had reversed the encoding for Sex?

# encode Sex_Male feature
titanic['Sex_Male'] = titanic.Sex.map({'male':1, 'female':0})

# include Sex_Male in the model instead of Sex_Female
feature_cols = ['Pclass', 'Parch', 'Age', 'Sex_Male']
X = titanic[feature_cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
logreg.fit(X_train, y_train)
zip(feature_cols, logreg.coef_[0])

# The coefficient is the same, except that it's **negative instead of positive**. As such, your choice of category for the baseline does not matter, all that changes is your **interpretation** of the coefficient.

# ### Dummy encoding with more than two levels

# How do we include an unordered categorical feature with more than two levels, like **Embarked**? We can't simply encode it as C=1, Q=2, S=3, because that would imply an **ordered relationship** in which Q is somehow "double" C and S is somehow "triple" C.
# Instead, we create **additional dummy variables**:

# create 3 dummy variables
pd.get_dummies(titanic.Embarked, prefix='Embarked').head(10)

# However, we actually only need **two dummy variables, not three**. Why? Because two dummies captures all of the "information" about the Embarked feature, and implicitly defines C as the **baseline level**.

# create 3 dummy variables, then exclude the first
pd.get_dummies(titanic.Embarked, prefix='Embarked').iloc[:, 1:].head(10)

# Here is how we interpret the encoding:
# - C is encoded as Embarked_Q=0 and Embarked_S=0
# - Q is encoded as Embarked_Q=1 and Embarked_S=0
# - S is encoded as Embarked_Q=0 and Embarked_S=1

# If this is confusing, think about why we only needed one dummy variable for Sex (Sex_Female), not two dummy variables (Sex_Female and Sex_Male). In general, if you have a categorical feature with **k levels**, you create **k-1 dummy variables**.

# create a DataFrame with the two dummy variable columns
embarked_dummies = pd.get_dummies(titanic.Embarked, prefix='Embarked').iloc[:, 1:]

# concatenate the original DataFrame and the dummy DataFrame (axis=0 means rows, axis=1 means columns)
titanic = pd.concat([titanic, embarked_dummies], axis=1)
titanic.head()

# include Embarked_Q and Embarked_S in the model
feature_cols = ['Pclass', 'Parch', 'Age', 'Sex_Female', 'Embarked_Q', 'Embarked_S']
X = titanic[feature_cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
logreg=LogisticRegression(C=1e9)
logreg.fit(X_train, y_train)
zip(feature_cols, logreg.coef_[0])

# How do we interpret the Embarked coefficients? They are **measured against the baseline (C)**, and thus embarking at Q is associated with a decrease in the likelihood of survival compared with C, and embarking at S is associated with a further decrease in the likelihood of survival.


# ## Part 4: ROC Curves and AUC

# predict probability of survival
y_pred_prob = logreg.predict_proba(X_test)[:, 1]

# plot ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

# Besides allowing you to calculate AUC, seeing the ROC curve can help you to choose a threshold that **balances sensitivity and specificity** in a way that makes sense for the particular context.

# calculate AUC
print metrics.roc_auc_score(y_test, y_pred_prob)

# It's important to use **y_pred_prob** and not **y_pred_class** when computing an ROC curve or AUC. If you use y_pred_class, it will not give you an error, rather it will interpret the ones and zeros as predicted probabilities of 100% and 0%, and thus will give you incorrect results:

# calculate AUC using y_pred_class (producing incorrect results)
print metrics.roc_auc_score(y_test, y_pred_class)

# histogram of predicted probabilities grouped by actual response value
df = pd.DataFrame(data = {'probability':y_pred_prob, 'actual':y_test})
df.probability.hist(by=df.actual, sharex=True, sharey=True)
