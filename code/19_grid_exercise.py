'''
EXERCISE: GridSearchCV with Stack Overflow competition data
'''

import pandas as pd

# define a function to create features
def make_features(filename):
    df = pd.read_csv(filename, index_col=0)
    df.rename(columns={'OwnerUndeletedAnswerCountAtPostTime':'Answers'}, inplace=True)
    df['TitleLength'] = df.Title.apply(len)
    df['BodyLength'] = df.BodyMarkdown.apply(len)
    df['NumTags'] = df.loc[:, 'Tag1':'Tag5'].notnull().sum(axis=1)
    return df

# apply function to both training and testing files
train = make_features('train.csv')
test = make_features('test.csv')

# define X and y
feature_cols = ['ReputationAtPostCreation', 'Answers', 'TitleLength', 'BodyLength', 'NumTags']
X = train[feature_cols]
y = train.OpenStatus


'''
MAIN TASK: Use GridSearchCV to find optimal parameters for KNeighborsClassifier.
- For 'n_neighbors', try 5 different integer values.
- For 'weights', try 'uniform' and 'distance'.
- Use 5-fold cross-validation (instead of 10-fold) to save computational time.
- Remember that log loss is your evaluation metric!

BONUS TASK #1: Once you have found optimal parameters, train your KNN model using
those parameters, make predictions on the test set, and submit those predictions.

BONUS TASK #2: Read the scikit-learn documentation for GridSearchCV to find the
shortcut for accomplishing bonus task #1.
'''

# MAIN TASK
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
from sklearn.grid_search import GridSearchCV
neighbors_range = [20, 40, 60, 80, 100]
weight_options = ['uniform', 'distance']
param_grid = dict(n_neighbors=neighbors_range, weights=weight_options)
grid = GridSearchCV(knn, param_grid, cv=5, scoring='log_loss')
grid.fit(X, y)
grid.grid_scores_
grid.best_score_
grid.best_params_

# BONUS TASK #1
knn = KNeighborsClassifier(n_neighbors=100, weights='uniform')
knn.fit(X, y)
X_oos = test[feature_cols]
oos_pred_prob = knn.predict_proba(X_oos)[:, 1]
sub = pd.DataFrame({'id':test.index, 'OpenStatus':oos_pred_prob}).set_index('id')
sub.to_csv('sub.csv')

# BONUS TASK #2
oos_pred_prob = grid.predict_proba(X_oos)[:, 1]
sub = pd.DataFrame({'id':test.index, 'OpenStatus':oos_pred_prob}).set_index('id')
sub.to_csv('sub.csv')
