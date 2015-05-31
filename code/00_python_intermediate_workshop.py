'''
LISTS
'''

# creating
a = [1, 2, 3, 4, 5]     # create lists using brackets

# slicing
a[0]        # returns 1 (Python is zero indexed)
a[1:3]      # returns [2, 3] (inclusive of first index but exclusive of second)
a[-1]       # returns 5 (last element)

# appending
a[5] = 6        # error because you can't assign outside the existing range
a.append(6)     # list method that appends 6 to the end
a = a + [7]     # use plus sign to combine lists

# sorting
sorted(a)               # sorts the list
sorted(a, reverse=True) # reverse=True is an 'optional argument'
sorted(a, True)         # error because optional arguments must be named

# checking type
type(a)     # returns list
type(a[0])  # returns int

# checking length
len(a)      # returns 7


'''
STRINGS
'''

# creating
a = 'hello'     # can use single or double quotes

# slicing
a[0]        # returns 'h' (works like list slicing)
a[1:3]      # returns 'el'
a[-1]       # returns 'o'

# concatenating
a + ' there'        # use plus sign to combine strings
5 + ' there'        # error because they are different types
str(5) + ' there'   # cast 5 to a string in order for this to work

# uppercasing
a[0] = 'H'      # error because strings are immutable (can't overwrite characters)
a.upper()       # string method (this method doesn't exist for lists)

# checking length
len(a)      # returns 5 (number of characters)


'''
FOR LOOPS AND LIST COMPREHENSIONS
'''

# for loop to print 1 through 5
nums = range(1, 6)      # create a list of 1 through 5
for num in nums:        # num 'becomes' each list element for one loop
    print num

# for loop to print 1, 3, 5
other = [1, 3, 5]       # create a different list
for x in other:         # name 'x' does not matter
    print x             # this loop only executes 3 times (not 5)

# for loop to create a list of cubes of 1 through 5
cubes = []                  # create empty list to store results
for num in nums:            # loop through nums (will execute 5 times)
    cubes.append(num**3)    # append the cube of the current value of num

# equivalent list comprehension
cubes = [num**3 for num in nums]    # expression (num**3) goes first, brackets
                                    # indicate we are storing results in a list


'''
EXERCISE:
Given that: letters = ['a','b','c']
Write a list comprehension that returns: ['A','B','C']

BONUS EXERCISE:
Given that: word = 'abc'
Write a list comprehension that returns: ['A','B','C']
'''

letters = ['a', 'b', 'c']
[letter.upper() for letter in letters]  # iterate through a list of strings,
                                        # and each string has an 'upper' method
word = 'abc'
[letter.upper() for letter in word]     # iterate through each character


'''
DICTIONARIES

dictionaries are similar to lists:
- both can contain multiple data types
- both are iterable
- both are mutable

dictionaries are different from lists:
- dictionaries are unordered
- dictionary lookup time is constant regardless of dictionary size

dictionaries are like real dictionaries:
- dictionaries are made of key-value pairs (word and definition)
- dictionary keys must be unique (each word is only defined once)
- you can use the key to look up the value, but not the other way around
'''

# create a dictionary
family = {'dad':'Homer', 'mom':'Marge', 'size':2}

# examine a dictionary
family[0]           # error because there is no ordering
family['dad']       # returns 'Homer' (use a key to look up a value)
len(family)         # returns 3 (number of key-value pairs)
family.keys()       # returns list: ['dad', 'mom', 'size']
family.values()     # returns list: ['Homer', 'Marge', 2]
family.items()      # returns list of tuples:
                    #   [('dad', 'Homer'), ('mom', 'Marge'), ('size', 2)]

# modify a dictionary
family['cat'] = 'snowball'          # add a new entry
family['cat'] = 'snowball ii'       # edit an existing entry
del family['cat']                   # delete an entry
family['kids'] = ['bart', 'lisa']   # value can be a list

# accessing a list element within a dictionary
family['kids'][0]   # returns 'bart'


'''
EXERCISE:
Print the name of the mom.
Change the size to 5.
Add 'Maggie' to the list of kids.
Fix 'bart' and 'lisa' so that the first letter is capitalized.
'''

family['mom']                       # returns 'Marge'
family['size'] = 5                  # replaces existing value for 'size'
family['kids'].append('Maggie')     # access a list, then append 'Maggie' to it
family['kids'][0] = 'Bart'          # capitalize names by overwriting them
family['kids'][1] = 'Lisa'

# or, capitalize using a list comprehension and the 'capitalize' string method
family['kids'] = [kid.capitalize() for kid in family['kids']]

# or, slice the string, uppercase the first letter, and concatenate with other letters
family['kids'] = [kid[0].upper()+kid[1:] for kid in family['kids']]


'''
REQUESTS
'''

import requests     # import module (make its functions available)

# use requests to talk to the web
r = requests.get('http://www.google.com')
type(r)         # special 'response' object
r.text          # HTML of web page stored as string
type(r.text)    # string is encoded as unicode
r.text[0]       # string can be sliced like any string


'''
APIs

API Providers: https://apigee.com/providers
Echo Nest API Console: https://apigee.com/console/echonest
Echo Nest Developer Center (for obtaining API key): http://developer.echonest.com/
'''

# request data from the Echo Nest API
r = requests.get('http://developer.echonest.com/api/v4/artist/top_hottt?api_key=KBGUPZPJZS9PHWNIN&format=json')
r.text          # looks like a dictionary
type(r.text)    # actually stored as a string
r.json()        # decodes JSON
type(r.json())  # JSON can be represented as a dictionary
top = r.json()  # store that dictionary

# store the artist data
artists = top['response']['artists']    # list of 15 dictionaries

# create a list of artist names only
names = [artist['name'] for artist in artists]  # can iterate through list to access dictionaries


'''
WORKING WITH PUBLIC DATA

FiveThirtyEight: http://fivethirtyeight.com/
FiveThirtyEight data: https://github.com/fivethirtyeight/data
NFL ticket prices data: https://github.com/fivethirtyeight/data/tree/master/nfl-ticket-prices

Question: What is the average ticket price for Ravens' home vs away games, and
how do those prices compare to the overall average ticket price?
'''

# open a CSV file from a URL (ignore any warnings) and store in a list of lists
import csv
r = requests.get('https://raw.githubusercontent.com/fivethirtyeight/data/master/nfl-ticket-prices/2014-average-ticket-price.csv')
data = [row for row in csv.reader(r.iter_lines())]

# or, open a downloaded CSV file from your working directory
with open('2014-average-ticket-price.csv', 'rU') as f:
    data = [row for row in csv.reader(f)]

# examine the data
type(data)      # list
len(data)       # every list represents a row in the CSV file
data[0]         # header row (list)
data[1]         # first row of data (list)

# only save the data we want
data = data[1:97]

# step 1: create a list that only contains events
events = [row[0] for row in data]       # grab the first element of each row

# step 2: create a list that only contains prices (stored as integers)
prices = [int(row[2]) for row in data]  # cast to an int before storing the price

'''
Optionally, complete this exercise at home and send me your code! There are
many ways to solve this, but here is the sequence of steps I would use:

Step 3: Calculate the overall average ticket price.
Hint: Calculate the sum of the list and divide by the length, and keep in mind
      that one of the numbers must be a float in order to perform "real" division.

Step 4: Use a for loop to make a list of the away teams.
Hint: Use the string method 'find' to locate the end of the away team name,
      and then slice the string up to that point.

Step 5: Use a for loop to make a list of the home teams.
Hint: Use the string method 'find' to locate the start and end of the home team
      name, and then slice the string.

Step 6: Create a list of prices for Ravens home games.
Hint: Use the zip function to pair home teams and prices, use a for loop to
      iterate through both home teams and prices at once, and then add a condition
      to only save the price if the team is the Ravens.

Step 7: Create a list of prices for Ravens away games.

Step 8: Calculate the average of each of the price lists.
'''
