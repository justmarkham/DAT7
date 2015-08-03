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
