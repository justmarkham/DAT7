'''
EXERCISE: "Human Learning" with iris data

Can you predict the species of an iris using petal and sepal measurements?

TASKS:
1. Read the iris data into a pandas DataFrame, including column names.
2. Gather some basic information about the data.
3. Use groupby, sorting, and plotting to look for differences between species.
4. Write down a set of rules that could be used to predict species based on measurements.

BONUS: Define a function that accepts a row of data and returns a predicted species.
Then, use that function to make predictions for all existing rows of data.
'''

import pandas as pd

## TASK 1

# read the iris data into a pandas DataFrame, including column names
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                   header=None, names=col_names)

## TASK 2

# gather basic information
iris.shape
iris.head()
iris.describe()
iris.species.value_counts()
iris.dtypes
iris.isnull().sum()

## TASK 3

# use groupby to look for differences between the species
iris.groupby('species').sepal_length.mean()
iris.groupby('species').mean()
iris.groupby('species').describe()

# use sorting to look for differences between the species
iris.sort('sepal_length').values
iris.sort('sepal_width').values
iris.sort('petal_length').values
iris.sort('petal_width').values

# use plotting to look for differences between the species
iris.petal_width.hist(by=iris.species, sharex=True)
iris.boxplot(column='petal_width', by='species')
iris.boxplot(by='species')

# map species to a numeric value so that plots can be colored by category
iris['species_num'] = iris.species.map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})
iris.plot(kind='scatter', x='petal_length', y='petal_width', c='species_num', colormap='Blues')
pd.scatter_matrix(iris, c=iris.species_num, figsize=(10, 8))

## TASK 4

# If petal length is less than 3, predict setosa.
# Else if petal width is less than 1.8, predict versicolor.
# Otherwise predict virginica.

## BONUS

# define a function that accepts a row of data and returns a predicted species
def classify_iris(row):
    if row[2] < 3:          # petal_length
        return 0    # setosa
    elif row[3] < 1.8:      # petal_width
        return 1    # versicolor
    else:
        return 2    # virginica

# predict for a single row to test the function
classify_iris(iris.iloc[0, :])      # first row
classify_iris(iris.iloc[149, :])    # last row

# store predictions for all rows
predictions = [classify_iris(row) for row in iris.values]

# calculate the percentage of correct predictions
import numpy as np
np.mean(iris.species_num == predictions)    # 0.96
