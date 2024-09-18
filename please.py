import pandas as pd

#
#  here is a Python list:

a = [1, 2, 3, 4, 5, 6]

# get a list containing the last 3 elements of a
# Yes, you can just type out [4, 5, 6] but we really want to see you demonstrate you know how to use list slicing in Python

b = a[3::]
print(b,'\n')

# create a list of numbers from 1 to 20
c = []
for i in range(20):
    c.append(i+1)
print(c, '\n')

# now get a list with only the even numbers between 1 and 100
# you may or may not make use of the list you made in the last cell

d = [i for i in range(1, 101)]
print(d,'\n')

# write a function that takes two numbers as arguments
# and returns the first number divided by the second

def division(num1, num2):
    if num2 != 0:
        return num1/num2
    

# fizzbuzz
# you will need to use both iteration and control flow 
# go through all numbers from 1 to 30 in order
# if the number is a multiple of 3, print fizz
# if the number is a multiple of 5, print buzz
# if the number is a multiple of 3 and 5, print fizzbuzz and NOTHING ELSE
# if the number is neither a multiple of 3 nor a multiple of 5, print the number

for i in range(1, 31):
    if i % 3 == 0 and i % 5 == 0:
        print('fizzbuzz', i)
    elif i % 3 == 0:
        print('fizz', i)
    elif i % 5 == 0:
        print('buzz', i)
    else:
        continue

# create a dictionary that reflects the following menu pricing (taken from Ahmo's)
# Gyro: $9 
# Burger: $9
# Greek Salad: $8
# Philly Steak: $10

menu = {
    'Gyro': 9,
    'Burger': 9,
    'Greek Salad': 8,
    'Philly Steak': 10
}

# load in the "starbucks.csv" dataset
# refer to how we read the cereal.csv dataset in the tutorial

df = pd.read_csv('starbucks.csv')
# print(type(df))
# print(df.head(10))
# print(df.sample())
# print(df.describe())
# print(df.shape)
# print(df["beverage_category"])


# select all rows with more than and including 400 calories

print(df[df["calories"] >= 400])

# select all rows whose vitamin c content is higher than the iron content

df[df["vitamin c"] > df["iron"]]

# create a new column containing the caffeine per calories of each drink

df["caffein_per_calories"] = df["caffeine"] / df["calories"]
print(df.head())

# what is the average calorie across all items?

print(df["calories"].mean())

# how many different categories of beverages are there?

print(df["beverage_category"].nunique())

# what is the average # calories for each beverage category?

print(df.groupby('beverage_category')['calories'].mean())