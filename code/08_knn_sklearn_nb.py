# # K-nearest neighbors and scikit-learn

# ## Agenda
# - **K-nearest neighbors (KNN)**
#     - Review of the iris dataset
#     - Human learning on the iris dataset
#     - KNN classification
#     - Review of supervised learning
# - **scikit-learn**
#     - Requirements for working with data in scikit-learn
#     - scikit-learn's 4-step modeling pattern
#     - Tuning a KNN model
#     - Comparing KNN with other models

# ## Review of the iris dataset

# read the iris data into a DataFrame
import pandas as pd
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, names=col_names)
iris.head()

# ### Terminology
# - **150 observations** (n=150): each observation is one iris flower
# - **4 features** (p=4): sepal length, sepal width, petal length, and petal width
# - **Response**: iris species
# - **Classification problem** since response is categorical

# ## Human learning on the iris dataset

# How did we (as humans) predict the species for iris flowers?
# 1. We looked for features that seemed to correlate with the response.
# 2. We created a set of rules (using those features) to predict the species of an unknown iris.

# More generally:
# 1. We observed that the different species had (somewhat) dissimilar measurements.
# 2. We predicted the species for an unknown iris by:
#     - Looking for irises in the data with similar measurements
#     - Assuming that our unknown iris is the same species as those similar irises

# create a custom colormap
from matplotlib.colors import ListedColormap
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# map each iris species to a number
iris['species_num'] = iris.species.map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})

# create a scatter plot of PETAL LENGTH versus PETAL WIDTH and color by SPECIES
iris.plot(kind='scatter', x='petal_length', y='petal_width', c='species_num', colormap=cmap_bold)

# create a scatter plot of SEPAL LENGTH versus SEPAL WIDTH and color by SPECIES
iris.plot(kind='scatter', x='sepal_length', y='sepal_width', c='species_num', colormap=cmap_bold)

# ## K-nearest neighbors (KNN) classification

# 1. Pick a value for K.
# 2. Search for the K observations in the data that are "nearest" to the measurements of the unknown iris.
#     - Euclidian distance is often used as the distance metric, but other metrics are allowed.
# 3. Use the most popular response value from the K "nearest neighbors" as the predicted response value for the unknown iris.

# **Question:** What's the "best" value for K in this case?
# **Answer:** The value which produces the most accurate predictions on unseen data. We want to create a model that generalizes!

# ## Requirements for working with data in scikit-learn
# 1. Features and response are **separate objects**
# 2. Features and response should be entirely **numeric**
# 3. Features and response should be **NumPy arrays** (or easily convertible to NumPy arrays)
# 4. Features and response should have **specific shapes** (outlined below)

iris.head()

# store feature matrix in "X"
feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = iris[feature_cols]

# alternative ways to create "X"
X = iris.drop(['species', 'species_num'], axis=1)
X = iris.loc[:, 'sepal_length':'petal_width']
X = iris.iloc[:, 0:4]

# store response vector in "y"
y = iris.species_num

# check X's type
print type(X)
print type(X.values)

# check y's type
print type(y)
print type(y.values)

# check X's shape (n = number of observations, p = number of features)
print X.shape

# check y's shape (single dimension with length n)
print y.shape

# ## scikit-learn's 4-step modeling pattern

# **Step 1:** Import the class you plan to use
from sklearn.neighbors import KNeighborsClassifier

# **Step 2:** "Instantiate" the "estimator"
# - "Estimator" is scikit-learn's term for "model"
# - "Instantiate" means "make an instance of"
knn = KNeighborsClassifier(n_neighbors=1)

# - Name of the object does not matter
# - Can specify tuning parameters (aka "hyperparameters") during this step
# - All parameters not specified are set to their defaults
print knn

# **Step 3:** Fit the model with data (aka "model training")
# - Model is "learning" the relationship between X and y in our "training data"
# - Process through which learning occurs varies by model
# - Occurs in-place

knn.fit(X, y)

# - Once a model has been fit with data, it's called a "fitted model"

# **Step 4:** Predict the response for a new observation
# - New observations are called "out-of-sample" data
# - Uses the information it learned during the model training process
knn.predict([3, 5, 4, 2])

# - Returns a NumPy array, and we keep track of what the numbers "mean"
# - Can predict for multiple observations at once
X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
knn.predict(X_new)

# ## Tuning a KNN model

# instantiate the model (using the value K=5)
knn = KNeighborsClassifier(n_neighbors=5)

# fit the model with data
knn.fit(X, y)

# predict the response for new observations
knn.predict(X_new)

# calculate predicted probabilities of class membership
knn.predict_proba(X_new)

# print distances to nearest neighbors (and their identities)
knn.kneighbors([3, 5, 4, 2])

# ## Comparing KNN with other models

# Advantages of KNN:
# - Simple to understand and explain
# - Model training is fast
# - Can be used for classification and regression!

# Disadvantages of KNN:
# - Must store all of the training data
# - Prediction phase can be slow when n is large
# - Sensitive to irrelevant features
# - Sensitive to the scale of the data
# - Accuracy is (generally) not competitive with the best supervised learning methods
