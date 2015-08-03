'''
CLASS: Advanced scikit-learn
'''

'''
TASK: Searching for optimal parameters
FUNCTION: GridSearchCV
DOCUMENTATION: http://scikit-learn.org/stable/modules/grid_search.html
DATA: Titanic (n=891, p=5 selected, type=classification)
DATA DICTIONARY: https://www.kaggle.com/c/titanic-gettingStarted/data
'''

# read in and prepare titanic data
import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/DAT7/master/data/titanic.csv'
titanic = pd.read_csv(url)
titanic['Sex'] = titanic.Sex.map({'female':0, 'male':1})
titanic.Age.fillna(titanic.Age.mean(), inplace=True)
embarked_dummies = pd.get_dummies(titanic.Embarked, prefix='Embarked').iloc[:, 1:]
titanic = pd.concat([titanic, embarked_dummies], axis=1)

# define X and y
feature_cols = ['Pclass', 'Sex', 'Age', 'Embarked_Q', 'Embarked_S']
X = titanic[feature_cols]
y = titanic.Survived

# use cross-validation to find best max_depth
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score

# try max_depth=2
treeclf = DecisionTreeClassifier(max_depth=2, random_state=1)
cross_val_score(treeclf, X, y, cv=10, scoring='roc_auc').mean()

# try max_depth=3
treeclf = DecisionTreeClassifier(max_depth=3, random_state=1)
cross_val_score(treeclf, X, y, cv=10, scoring='roc_auc').mean()

# use GridSearchCV to automate the search
from sklearn.grid_search import GridSearchCV
treeclf = DecisionTreeClassifier(random_state=1)
max_depth_range = range(1, 21)
param_grid = dict(max_depth=max_depth_range)
grid = GridSearchCV(treeclf, param_grid, cv=10, scoring='roc_auc')
grid.fit(X, y)

# check the results of the grid search
grid.grid_scores_
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]

# plot the results
import matplotlib.pyplot as plt
plt.plot(max_depth_range, grid_mean_scores)
plt.xlabel('Value of max_depth')
plt.ylabel('Cross-Validated AUC')

# what was best?
grid.best_score_
grid.best_params_
grid.best_estimator_

# search multiple parameters simultaneously
max_depth_range = range(1, 21)
leaf_range = range(1, 11)
param_grid = dict(max_depth=max_depth_range, min_samples_leaf=leaf_range)
grid = GridSearchCV(treeclf, param_grid, cv=10, scoring='roc_auc')
grid.fit(X, y)
grid.grid_scores_
grid.best_score_
grid.best_params_


'''
TASK: Standardization of features (aka "center and scale" or "z-score normalization")
FUNCTION: StandardScaler
DOCUMENTATION: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
DATA: Wine (n=178, p=2 selected, type=classification)
DATA DICTIONARY: http://archive.ics.uci.edu/ml/datasets/Wine
'''

# fake data
train = pd.DataFrame({'id':[0,1,2], 'length':[0.9,0.3,0.6], 'mass':[0.1,0.2,0.8], 'rings':[40,50,60]})
test = pd.DataFrame({'length':[0.59], 'mass':[0.79], 'rings':[54.9]})

# define X and y
feature_cols = ['length', 'mass', 'rings']
X = train[feature_cols]
y = train.id

# KNN with k=1
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)

# what "should" it predict? what does it predict?
knn.predict(test)

# standardize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# compare original to standardized
X.values
X_scaled

# figure out how it standardized
scaler.mean_
scaler.std_
(X.values - scaler.mean_) / scaler.std_

# try this on real data
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
wine = pd.read_csv(url, header=None, usecols=[0, 10, 13])
wine.columns=['label', 'color', 'proline']
wine.head()
wine.describe()

# define X and y
feature_cols = ['color', 'proline']
X = wine[feature_cols]
y = wine.label

# split into train/test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# standardize X_train
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

# check that it worked properly
X_train_scaled[:, 0].mean()
X_train_scaled[:, 0].std()
X_train_scaled[:, 1].mean()
X_train_scaled[:, 1].std()

# standardize X_test
X_test_scaled = scaler.transform(X_test)

# is this right?
X_test_scaled[:, 0].mean()
X_test_scaled[:, 0].std()
X_test_scaled[:, 1].mean()
X_test_scaled[:, 1].std()

# KNN accuracy on original data
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_class = knn.predict(X_test)
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)

# KNN accuracy on scaled data
knn.fit(X_train_scaled, y_train)
y_pred_class = knn.predict(X_test_scaled)
metrics.accuracy_score(y_test, y_pred_class)


'''
TASK: Chaining steps
FUNCTION: Pipeline
DOCUMENTATION: http://scikit-learn.org/stable/modules/pipeline.html
DATA: Wine (n=178, p=2 selected, type=classification)
DATA DICTIONARY: http://archive.ics.uci.edu/ml/datasets/Wine
'''

# here is proper cross-validation on the original (unscaled) data
feature_cols = ['color', 'proline']
X = wine[feature_cols]
y = wine.label
knn = KNeighborsClassifier(n_neighbors=3)
cross_val_score(knn, X, y, cv=5, scoring='accuracy').mean()

# why is this improper cross-validation on the scaled data?
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy').mean()

# fix this using Pipeline
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()

# using GridSearchCV with Pipeline
neighbors_range = range(1, 21)
param_grid = dict(kneighborsclassifier__n_neighbors=neighbors_range)
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)
grid.best_score_
grid.best_params_
