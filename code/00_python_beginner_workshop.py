'''
Multi-line comments go between 3 quotation marks.
You can use single or double quotes.
'''

# One-line comments are preceded by the pound symbol


# BASIC DATA TYPES

x = 5               # creates an object
print type(x)       # check the type: int (not declared explicitly)
type(x)             # automatically prints
type(5)             # assigning it to a variable is not required

type(5.0)           # float
type('five')        # str
type(True)          # bool


# LISTS

nums = [5, 5.0, 'five']     # multiple data types
nums                        # print the list
type(nums)                  # check the type: list
len(nums)                   # check the length: 3
nums[0]                     # print first element
nums[0] = 6                 # replace a list element

nums.append(7)              # list 'method' that modifies the list
help(nums.append)           # help on this method
help(nums)                  # help on a list object
nums.remove('five')         # another list method

sorted(nums)                # 'function' that does not modify the list
nums                        # it was not affected
nums = sorted(nums)         # overwrite the original list
sorted(nums, reverse=True)  # optional argument

# list slicing [start:end:stride]
weekdays = ['mon','tues','wed','thurs','fri']
weekdays[0]         # element 0
weekdays[0:3]       # elements 0, 1, 2
weekdays[:3]        # elements 0, 1, 2
weekdays[3:]        # elements 3, 4
weekdays[-1]        # last element (element 4)
weekdays[::2]       # every 2nd element (0, 2, 4)
weekdays[::-1]      # backwards (4, 3, 2, 1, 0)

days = weekdays + ['sat','sun']     # concatenate lists


# FUNCTIONS

def give_me_five():         # function definition ends with colon
    return 5                # indentation required for function body

give_me_five()              # prints the return value (5)
num = give_me_five()        # assigns return value to a variable, doesn't print it

def calc(x, y, op):         # three parameters (without any defaults)
    if op == 'add':         # conditional statement
        return x + y
    elif op == 'subtract':
        return x - y
    else:
        print 'Valid operations: add, subtract'

calc(5, 3, 'add')
calc(5, 3, 'subtract')
calc(5, 3, 'multiply')
calc(5, 3)


# EXERCISE: Write a function that takes two parameters (hours and rate), and
# returns the total pay.

def compute_pay(hours, rate):
    return hours * rate

compute_pay(40, 10.50)


# FOR LOOPS

# range returns a list of integers
range(0, 3)     # returns [0, 1, 2]: includes first value but excludes second value
range(3)        # same thing: starting at zero is the default

# simple for loop
for i in range(5):
    print i

# print each list element in uppercase
fruits = ['apple', 'banana', 'cherry']
for i in range(len(fruits)):
    print fruits[i].upper()

# better for loop
for fruit in fruits:
    print fruit.upper()


# EXERCISE: Write a program that prints the numbers from 1 to 100. But for
# multiples of 3 print 'fizz' instead of the number, and for the multiples of
# 5 print 'buzz'. For numbers which are multiples of both 3 and 5 print 'fizzbuzz'.

def fizz_buzz():
    nums = range(1, 101)    
    for num in nums:
        if num % 15 == 0:
            print 'fizzbuzz'
        elif num % 3 == 0:
            print 'fizz'
        elif num % 5 == 0:
            print 'buzz'
        else:
            print num

fizz_buzz()
