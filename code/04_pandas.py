'''
CLASS: Pandas for Exploratory Data Analysis

MovieLens 100k movie rating data:
    main page: http://grouplens.org/datasets/movielens/
    data dictionary: http://files.grouplens.org/datasets/movielens/ml-100k-README.txt
    files: u.user, u.data, u.item

WHO alcohol consumption data:
    article: http://fivethirtyeight.com/datalab/dear-mona-followup-where-do-people-drink-the-most-beer-wine-and-spirits/    
    original data: https://github.com/fivethirtyeight/data/tree/master/alcohol-consumption
    file: drinks.csv (with additional 'continent' column)

National UFO Reporting Center data:
    main page: http://www.nuforc.org/webreports.html
    file: ufo.csv
'''

import pandas as pd

'''
Reading Files, Selecting Columns, and Summarizing
'''

# can read a file from local computer or directly from a URL
pd.read_table('u.user')
pd.read_table('https://raw.githubusercontent.com/justmarkham/DAT7/master/data/u.user')

# read 'u.user' into 'users'
user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users = pd.read_table('u.user', sep='|', header=None, names=user_cols, index_col='user_id', dtype={'zip_code':str})

# examine the users data
users                   # print the first 30 and last 30 rows
type(users)             # DataFrame
users.head()            # print the first 5 rows
users.head(10)          # print the first 10 rows
users.tail()            # print the last 5 rows
users.index             # "the index" (aka "the labels")
users.columns           # column names (which is "an index")
users.dtypes            # data types of each column
users.shape             # number of rows and columns
users.values            # underlying numpy array
users.info()            # concise summary (including memory usage)

# select a column
users['gender']         # select one column
type(users['gender'])   # Series
users.gender            # select one column using the DataFrame attribute

# summarize (describe) the data
users.describe()                    # describe all numeric columns
users.describe(include=['object'])  # describe all object columns (can include multiple types)
users.describe(include='all')       # describe all columns
users.gender.describe()             # describe a single column
users.age.mean()                    # only calculate the mean

# count the number of occurrences of each value
users.gender.value_counts()     # most useful for categorical variables
users.age.value_counts()        # can also be used with numeric variables

'''
EXERCISE ONE
'''

# read drinks.csv into a DataFrame called 'drinks'
drinks = pd.read_table('drinks.csv', sep=',')
drinks = pd.read_csv('drinks.csv')              # assumes separator is comma

# print the head and the tail
drinks.head()
drinks.tail()

# examine the default index, data types, and shape
drinks.index
drinks.dtypes
drinks.shape

# print the 'beer_servings' Series
drinks['beer_servings']
drinks.beer_servings

# calculate the average 'beer_servings' for the entire dataset
drinks.describe()                   # summarize all numeric columns
drinks.beer_servings.describe()     # summarize only the 'beer_servings' Series
drinks.beer_servings.mean()         # only calculate the mean

# count the number of occurrences of each 'continent' value and see if it looks correct
drinks.continent.value_counts()

'''
Filtering and Sorting
'''

# logical filtering: only show users with age < 20
young_bool = users.age < 20         # create a Series of booleans...
users[young_bool]                   # ...and use that Series to filter rows
users[users.age < 20]               # or, combine into a single step
users[users.age < 20].occupation    # select one column from the filtered results
users[users.age < 20].occupation.value_counts()     # value_counts of resulting Series

# logical filtering with multiple conditions
users[(users.age < 20) & (users.gender=='M')]       # ampersand for AND condition
users[(users.age < 20) | (users.age > 60)]          # pipe for OR condition
users[users.occupation.isin(['doctor', 'lawyer'])]  # alternative to multiple OR conditions

# sorting
users.age.order()                   # sort a column
users.sort('age')                   # sort a DataFrame by a single column
users.sort('age', ascending=False)  # use descending order instead
users.sort(['occupation', 'age'])   # sort by multiple columns

'''
EXERCISE TWO
'''

# filter DataFrame to only include European countries
drinks[drinks.continent=='EU']

# filter DataFrame to only include European countries with wine_servings > 300
drinks[(drinks.continent=='EU') & (drinks.wine_servings > 300)]

# calculate the average 'beer_servings' for all of Europe
drinks[drinks.continent=='EU'].beer_servings.mean()

# determine which 10 countries have the highest total_litres_of_pure_alcohol
drinks.sort('total_litres_of_pure_alcohol').tail(10)

'''
Renaming, Adding, and Removing Columns
'''

# renaming one or more columns
drinks.rename(columns={'beer_servings':'beer', 'wine_servings':'wine'})
drinks.rename(columns={'beer_servings':'beer', 'wine_servings':'wine'}, inplace=True)

# replace all column names
drink_cols = ['country', 'beer', 'spirit', 'wine', 'liters', 'continent']
drinks = pd.read_csv('drinks.csv', header=0, names=drink_cols)  # replace during file reading
drinks.columns = drink_cols                                     # replace after file reading

# add a new column as a function of existing columns
drinks['servings'] = drinks.beer + drinks.spirit + drinks.wine
drinks['mL'] = drinks.liters * 1000

# removing columns
drinks.drop('mL', axis=1)                               # axis=0 for rows, 1 for columns
drinks.drop(['mL', 'servings'], axis=1)                 # drop multiple columns
drinks.drop(['mL', 'servings'], axis=1, inplace=True)   # make it permanent

'''
Handling Missing Values
'''

# missing values are usually excluded by default
drinks.continent.value_counts()              # excludes missing values
drinks.continent.value_counts(dropna=False)  # includes missing values

# find missing values in a Series
drinks.continent.isnull()           # True if missing, False if not missing
drinks.continent.isnull().sum()     # count the missing values
drinks.continent.notnull()          # True if not missing, False if missing
drinks[drinks.continent.notnull()]  # only show rows where continent is not missing

# side note: understanding axes
drinks.sum(axis=0)      # sums "down" the 0 axis (rows)
drinks.sum()            # axis=0 is the default
drinks.sum(axis=1)      # sums "across" the 1 axis (columns)

# find missing values in a DataFrame
drinks.isnull()             # DataFrame of booleans
drinks.isnull().sum()       # count the missing values in each column

# drop missing values
drinks.dropna()             # drop a row if ANY values are missing
drinks.dropna(how='all')    # drop a row only if ALL values are missing

# fill in missing values
drinks.continent.fillna(value='NA')                 # fill in missing values with 'NA'
drinks.continent.fillna(value='NA', inplace=True)   # modifies 'drinks' in-place

# turn off the missing value filter
drinks = pd.read_csv('drinks.csv', header=0, names=drink_cols, na_filter=False)

'''
EXERCISE THREE
'''

# read ufo.csv into a DataFrame called 'ufo'

# check the shape of the DataFrame

# what are the three most common colors reported?

# rename any columns with spaces so that they don't contain spaces

# for reports in VA, what's the most common city?

# print a DataFrame containing only reports from Arlington, VA

# count the number of missing values in each column

# how many rows remain if you drop all rows with any missing values?

'''
Split-Apply-Combine
Diagram: http://i.imgur.com/yjNkiwL.png
'''

# for each continent, calculate the mean beer servings
drinks.groupby('continent').beer.mean()

# for each continent, calculate the mean of all numeric columns
drinks.groupby('continent').mean()

# for each continent, describe beer servings
drinks.groupby('continent').beer.describe()

# similar, but outputs a DataFrame and can be customized
drinks.groupby('continent').beer.agg(['count', 'mean', 'min', 'max'])
drinks.groupby('continent').beer.agg(['count', 'mean', 'min', 'max']).sort('mean')

# for each continent, describe all numeric columns
drinks.groupby('continent').describe()

# for each continent, count the number of occurrences
drinks.groupby('continent').continent.count()
drinks.continent.value_counts()

'''
EXERCISE FOUR
'''

# for each occupation in 'users', count the number of occurrences

# for each occupation, calculate the mean age

# for each occupation, calculate the minimum and maximum ages

# for each combination of occupation and gender, calculate the mean age

'''
Selecting Multiple Columns and Filtering Rows
'''

# select multiple columns
my_cols = ['City', 'State']     # create a list of column names...
ufo[my_cols]                    # ...and use that list to select columns
ufo[['City', 'State']]          # or, combine into a single step

# use loc to select columns by name
ufo.loc[:, 'City']              # colon means "all rows", then select one column
ufo.loc[:, ['City', 'State']]   # select two columns
ufo.loc[:, 'City':'State']      # select a range of columns

# loc can also filter rows by "name" (the index)
ufo.loc[0, :]                   # row 0, all columns
ufo.loc[0:2, :]                 # rows 0/1/2, all columns
ufo.loc[0:2, 'City':'State']    # rows 0/1/2, range of columns

# use iloc to filter rows and select columns by integer position
ufo.iloc[:, [0, 3]]             # all rows, columns in position 0/3
ufo.iloc[:, 0:4]                # all rows, columns in position 0/1/2/3
ufo.iloc[0:3, :]                # rows in position 0/1/2, all columns

'''
Joining (Merging) DataFrames
'''

# read 'u.item' into 'movies'
movie_cols = ['movie_id', 'title']
movies = pd.read_table('u.item', sep='|', header=None, names=movie_cols, usecols=[0, 1])

# read 'u.data' into 'ratings'
rating_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('u.data', sep='\t', header=None, names=rating_cols)

# merge 'movies' and 'ratings' (inner join on 'movie_id')
movie_ratings = pd.merge(movies, ratings)
movies.shape
ratings.shape
movie_ratings.shape

'''
Other Commonly Used Features
'''

# map existing values to a different set of values
users['is_male'] = users.gender.map({'F':0, 'M':1})

# replace all instances of a value in a column (must match entire value)
ufo.State.replace('Fl', 'FL', inplace=True)

# string methods are accessed via 'str'
ufo.State.str.upper()                               # converts to uppercase
ufo.Colors_Reported.str.contains('RED', na='False') # checks for a substring

# convert a string to the datetime format
ufo['Time'] = pd.to_datetime(ufo.Time)
ufo.Time.dt.hour                        # datetime format exposes convenient attributes
(ufo.Time.max() - ufo.Time.min()).days  # also allows you to do datetime "math"

# setting and then removing an index
ufo.set_index('Time', inplace=True)
ufo.reset_index(inplace=True)

# change the data type of a column
drinks['beer'] = drinks.beer.astype('float')

# create dummy variables for 'continent' and exclude first dummy column
continent_dummies = pd.get_dummies(drinks.continent, prefix='cont').iloc[:, 1:]

# concatenate two DataFrames (axis=0 for rows, axis=1 for columns)
drinks = pd.concat([drinks, continent_dummies], axis=1)

'''
Other Less Used Features
'''

# detecting duplicate rows
users.duplicated()          # True if a row is identical to a previous row
users.duplicated().sum()    # count of duplicates
users[users.duplicated()]   # only show duplicates
users.drop_duplicates()     # drop duplicate rows
users.age.duplicated()      # check a single column for duplicates
users.duplicated(['age', 'gender', 'zip_code']).sum()   # specify columns for finding duplicates

# convert a range of values into descriptive groups
drinks['beer_level'] = 'low'    # initially set all values to 'low'
drinks.loc[drinks.beer.between(101, 200), 'beer_level'] = 'med'     # change 101-200 to 'med'
drinks.loc[drinks.beer.between(201, 400), 'beer_level'] = 'high'    # change 201-400 to 'high'

# display a cross-tabulation of two Series
pd.crosstab(drinks.continent, drinks.beer_level)

# convert 'beer_level' into the 'category' data type
drinks['beer_level'] = pd.Categorical(drinks.beer_level, categories=['low', 'med', 'high'])
drinks.sort('beer_level')   # sorts by the categorical ordering (low to high)

# limit which rows are read when reading in a file
pd.read_csv('drinks.csv', nrows=10)           # only read first 10 rows
pd.read_csv('drinks.csv', skiprows=[1, 2])    # skip the first two rows of data

# write a DataFrame out to a CSV
drinks.to_csv('drinks_updated.csv')                 # index is used as first column
drinks.to_csv('drinks_updated.csv', index=False)    # ignore index

# create a DataFrame from a dictionary
pd.DataFrame({'capital':['Montgomery', 'Juneau', 'Phoenix'], 'state':['AL', 'AK', 'AZ']})

# create a DataFrame from a list of lists
pd.DataFrame([['Montgomery', 'AL'], ['Juneau', 'AK'], ['Phoenix', 'AZ']], columns=['capital', 'state'])

# randomly sample a DataFrame
import numpy as np
mask = np.random.rand(len(drinks)) < 0.66   # create a Series of booleans
train = drinks[mask]                        # will contain around 66% of the rows
test = drinks[~mask]                        # will contain the remaining rows

# change the maximum number of rows and columns printed ('None' means unlimited)
pd.set_option('max_rows', None)     # default is 60 rows
pd.set_option('max_columns', None)  # default is 20 columns
print drinks

# reset options to defaults
pd.reset_option('max_rows')
pd.reset_option('max_columns')

# change the options temporarily (settings are restored when you exit the 'with' block)
with pd.option_context('max_rows', None, 'max_columns', None):
    print drinks
