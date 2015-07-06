'''
HOMEWORK: Yelp Reviews
'''

# TASK 1: read the data from yelp.csv into a DataFrame

import pandas as pd
yelp = pd.read_csv('yelp.csv')


# TASK 1 (ALTERNATIVE): construct the same DataFrame from yelp.json

# read the data from yelp.json into a list of rows
# each row is decoded into a dictionary using using json.loads()
import json
with open('yelp.json', 'rU') as f:
    data = [json.loads(row) for row in f]

# convert the list of dictionaries to a DataFrame
yelp = pd.DataFrame(data)

# add columns for cool, useful, and funny
yelp['cool'] = [row['votes']['cool'] for row in data]
yelp['useful'] = [row['votes']['useful'] for row in data]
yelp['funny'] = [row['votes']['funny'] for row in data]

# drop the votes column
yelp.drop('votes', axis=1, inplace=True)


# TASK 2: explore the relationship between cool/useful/funny and stars

# treat stars as a categorical variable and look for differences between groups
yelp.groupby('stars').mean()

# correlation matrix
import seaborn as sns
sns.heatmap(yelp.corr())

# scatter plot matrix
sns.pairplot(yelp, kind='reg')

# limit scatter plot matrix and add regression lines
sns.pairplot(yelp, x_vars=['cool', 'useful', 'funny'], y_vars='stars', size=6, aspect=0.7, kind='reg')


# TASK 3: define cool/useful/funny as the features and stars as the response

feature_cols = ['cool', 'useful', 'funny']
X = yelp[feature_cols]
y = yelp.stars


# TASK 4: fit a linear regression model and interpret the coefficients

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X, y)
zip(feature_cols, linreg.coef_)


# TASK 5: use train/test split and RMSE to evaluate the model

from sklearn.cross_validation import train_test_split
from sklearn import metrics
import numpy as np

def train_test_rmse(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))

train_test_rmse(X, y)


# TASK 6: try removing some of the features and see if RMSE improves

feature_cols = ['cool', 'funny']
X = yelp[feature_cols]
train_test_rmse(X, y)


# TASK 7 (BONUS): create new features, add them to the model, check RMSE

# new feature: review length (number of characters)
yelp['length'] = yelp.text.apply(len)

# new features: whether or not the review contains 'love' or 'hate'
yelp['love'] = yelp.text.str.contains('love', case=False).astype(int)
yelp['hate'] = yelp.text.str.contains('hate', case=False).astype(int)

# add new features to the model
feature_cols = ['cool', 'useful', 'funny', 'length', 'love', 'hate']
X = yelp[feature_cols]
train_test_rmse(X, y)


# TASK 8 (BONUS): compare your best RMSE with RMSE for the null model

# split the data (outside of the function)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# use scikit-learn's built-in dummy regressor
from sklearn.dummy import DummyRegressor
dumb = DummyRegressor(strategy='mean')
dumb.fit(X_train, y_train)
y_dumb = dumb.predict(X_test)
print np.sqrt(metrics.mean_squared_error(y_test, y_dumb))

# or, create a NumPy array with the right length, and fill it with the mean of y_train
y_null = np.zeros_like(y_test, dtype=float)
y_null.fill(y_train.mean())
print np.sqrt(metrics.mean_squared_error(y_test, y_null))


# TASK 9 (BONUS): treat this as a classification problem, try KNN, maximize your accuracy

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=150)
knn.fit(X_train, y_train)
y_pred_class = knn.predict(X_test)
print metrics.accuracy_score(y_test, y_pred_class)


# TASK 10 (BONUS): use linear regression for classification, and compare accuracy with KNN

# use linear regression to make continuous predictions
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)

# round its predictions to the nearest integer
y_pred_class = y_pred.round()

# compute classification accuracy of the rounded predictions
print metrics.accuracy_score(y_test, y_pred_class)
