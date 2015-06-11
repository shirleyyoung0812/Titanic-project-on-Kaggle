__author__ = 'shirleyyoung'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import csv
# always use header =  0 when row 0 is the header row
df = pd.read_csv('/Users/shirleyyoung/Documents/Kaggle/Titantic/train.csv', header = 0)
test_df = pd.read_csv('/Users/shirleyyoung/Documents/Kaggle/Titantic/test.csv', header = 0)

# add a new column
# .upper() upper case of the character
# df['Gender'] = df['Sex'].map(lambda x : x[0].upper())
# map female to 0 and male to 1
df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)


# fill in missing ages
# for each passenger without an age, fill the median age
# of his/her passenger class
median_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i, j] = df[(df['Gender'] == i) &
                               (df['Pclass'] == j + 1)]['Age'].dropna().median()

# create a new column to fill the missing age (for caution)
df['AgeFill'] = df['Age']
# since each column is a pandas data series object, the data cannot be accessed
# by df[2,3], we must provide the label (header) of the the column and use .loc()
# to locate the data e.g., df.loc[0, 'Age']
# or df[row]['header']
for i in range(2):
    for j in range(3):
        df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j + 1),
               'AgeFill'] = median_ages[i, j]

# fill the missing Embarked with the most common boarding place
# mode() returns the mode of the data set, which is the most frequent element in the data
# sometimes multiple values may be returned, thus in order to select the maximum
# use df.mode().iloc[0]
if len(df.Embarked[df.Embarked.isnull()]) > 0:
    df.Embarked[df.Embarked.isnull()] = df.Embarked.dropna().mode().iloc[0]

# returns an enumerate object
# e.g., [(0, 'S'), (1, 'C'),(2, 'Q')]
# Ports = list(enumerate(np.unique(df.Embarked)))
# Set up a dictionary that is an enumerate object of the ports
Ports_dict = {name : i for i, name in list(enumerate(np.unique(df.Embarked)))}
df['EmbarkFill'] = df.Embarked.map(lambda x: Ports_dict[x]).astype(int)

# create a column to indicate if the original age is null or not
# df['AgeIsNull'] = pd.isnull(df.Age).astype(int)


# create a column for family size
# df['FamilySize'] = df['SibSp'] + df['Parch']

# create a column that is the product of age and passenger class
# df['AgeClassProduct'] = df.AgeFill * df.Pclass

# drop the columns that we don't need
# axis = 1, column
df = df.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis = 1)


# do the same thing for test data
test_df = pd.read_csv('/Users/shirleyyoung/Documents/Kaggle/Titantic/test.csv', header=0)
test_df['Gender'] = test_df.Sex.map({'female': 0, 'male': 1}).astype(int)


test_df['AgeFill'] = test_df['Age']
for i in range(2):
    for j in range(3):
        test_df.loc[(test_df.Age.isnull()) & (test_df.Gender == i) & (test_df.Pclass == j + 1),
                    'AgeFill'] = median_ages[i, j]

if len(test_df.Embarked[test_df.Embarked.isnull()]) > 0:
    test_df.Embarked[test_df.Embarked.isnull()] = test_df.Embarked.dropna().mode().iloc[0]

test_df['EmbarkFill'] = test_df.Embarked.map(lambda x: Ports_dict[x]).astype(int)

# impute missing fares
if len(test_df.Fare[test_df.Fare.isnull()]) > 0:
    median_fare = np.zeros(3)
    for c in range(3):
        median_fare[c] = test_df[test_df.Pclass == c + 1]['Fare'].dropna().median()
    for c in range(3):
        test_df.loc[(test_df.Fare.isnull()) & (test_df.Pclass == c + 1),'Fare'] = median_fare[c]

# collect the test data's PassengerIds for output before dropping it
ids = test_df['PassengerId'].values
test_df = test_df.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis = 1)

# print(df.dtypes)
# print()
# print(test_df.dtypes)

# convert the data to numpy array
training_data = df.values
test_data = test_df.values


# train the data using random forest
# n_estimators: number of trees in the forest
forest = RandomForestClassifier(n_estimators=100)
# build the forest
# X: array-like or sparse matrix of shape = [n_samples, n_features]
# y: array-like, shape = [n_samples], target values/class labels
forest = forest.fit(training_data[0::, 1::], training_data[0::, 0])

output = forest.predict(test_data).astype(int)

# write the output to a new csv file
predictions_file = open("predictByRandomForest.csv", 'w')
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId", "Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()