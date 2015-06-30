'''
EXERCISE: Glass Identification (aka "Glassification")
'''

# TASK 1: read the data into a DataFrame
import pandas as pd
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data'
col_names = ['id','ri','na','mg','al','si','k','ca','ba','fe','glass_type']
df = pd.read_csv(url, names=col_names, index_col='id')

# TASK 2: briefly explore the data
df.shape
df.head()
df.tail()
df.glass_type.value_counts()
df.isnull().sum()

# TASK 3: convert into binary classification problem (see instructions for explanation)
df['binary'] = df.glass_type.map({1:0, 2:0, 3:0, 4:0, 5:1, 6:1, 7:1})
df.binary.value_counts()

# TASK 4: create a feature matrix (X) using all features (see instructions for explanation)
feature_cols = ['ri','na','mg','al','si','k','ca','ba','fe']
X = df[feature_cols]
X = df.drop(['glass_type','binary'], axis=1)    # alternative method

# TASK 5: create a response vector (y)
y = df.binary

# TASK 6: split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99)

# TASK 7: fit a KNN model on the training set using K=5
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# TASK 8: make predictions on the testing set and calculate testing accuracy
y_pred = knn.predict(X_test)
from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred)    # 90.7% accuracy

# TASK 9: write a for loop that computes testing accuracy for a range of K values
k_range = range(1, 30, 2)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    k_scores.append(metrics.accuracy_score(y_test, y_pred))

# TASK 10: plot K value versus testing accuracy to choose on optimal value for K
import matplotlib.pyplot as plt
plt.plot(k_range, k_scores)                     # optimal value is K=3

# TASK 11: calculate the null accuracy (see instructions for explanation)
1 - y_test.mean()                               # 74.1% null accuracy

# TASK 12: search for useful features
df.groupby('binary').mean()
df.boxplot(column='mg', by='binary')
df.boxplot(column='al', by='binary')
df.boxplot(column='ba', by='binary')

# redo exercise using only those features
feature_cols = ['mg','ba']
X = df[feature_cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    k_scores.append(metrics.accuracy_score(y_test, y_pred))
plt.plot(k_range, k_scores)                     # K=5 has 94.4% testing accuracy
