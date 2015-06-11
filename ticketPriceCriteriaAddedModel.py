__author__ = 'shirleyyoung'

import csv
import numpy as np

training_object = csv.reader(open('/Users/shirleyyoung/Documents/Kaggle/Titantic/train.csv', 'r'))
training_header = training_object.__next__()

# create a numpy multidimensional array object
data = []
for row in training_object:
    data.append(row)
data = np.array(data)



fare_ceiling = 40

# 0:: the 10th column, from row 0 to row last
# csv reader reads default to string,
# we need to convert it to float
# for ticket price higher than 39, it will be set to equal 39
# so that we can set 4 bins with equal size
# i.e., $0-9, $10-19, $20-29, $30-39
data[data[0::, 9].astype(np.float) >= fare_ceiling, 9] = fare_ceiling - 1.0

# basically make 4 equal bins
fare_bracket_size = 10
number_of_price_brackets = fare_ceiling / fare_bracket_size

# np.unique() return an array of unique elements in the object
# get the length of that array
number_of_classes = len(np.unique(data[0::, 2]))

# initialize the survival table with all zeros
survival_table = np.zeros((2, number_of_classes, number_of_price_brackets))

for i in range(number_of_classes):
    for j in range(int(number_of_price_brackets)):

        women_only_stats = data[(data[0::, 4] == "female")
                                & (data[0::, 2].astype(np.float) == i+1)  # i starts from 0,
                                # the ith class fare was greater than or equal to the least fare in current bin
                                & (data[0:, 9].astype(np.float) >= j*fare_bracket_size)
                                # fare was less than the least fare in next bin
                                & (data[0:, 9].astype(np.float) < (j+1)*fare_bracket_size), 1]

        men_only_stats = data[(data[0::, 4] != "female")
                              & (data[0::, 2].astype(np.float) == i + 1)
                              & (data[0:,9].astype(np.float) >= j * fare_bracket_size)
                              & (data[0:,9].astype(np.float) < (j + 1) * fare_bracket_size), 1]

        survival_table[0, i, j] = np.mean(women_only_stats.astype(np.float))
        survival_table[1, i, j] = np.mean(men_only_stats.astype(np.float))

# if nobody satisfies the criteria, the table will return a NaN
# since the divisor is zero
survival_table[survival_table != survival_table] = 0

# assume any probability >= 0.5 should result in predicting survival
# otherwise not
survival_table[survival_table < 0.5] = 0
survival_table[survival_table >= 0.5] = 1

test_file = open('/Users/shirleyyoung/Documents/Kaggle/Titantic/test.csv')
test_object = csv.reader(test_file)
test_header = test_object.__next__()
prediction_file = open("/Users/shirleyyoung/Documents/Kaggle/Titantic/genderClassModel.csv", 'w')
p = csv.writer(prediction_file)
p.writerow(["PassengerId", "Survived"])

# loop through each passenger
for row in test_object:
    # for each passenger, find the price bin where the passenger
    # belongs to
    try:
        row[8] = float(row[8])
    # if data is missing, bin the fare according Pclass
    except:
        bin_fare = 3 - float(row[1])
        continue
    # assign the passenger to the last bin if the fare he/she paid
    # was greater than the fare ceiling
    if row[8] > fare_ceiling:
        bin_fare = number_of_price_brackets - 1
    else:
        bin_fare = int(row[8] / fare_bracket_size)

    if row[3] == 'female':
        p.writerow([row[0], "%d" %
            int(survival_table[0, float(row[1]) - 1, bin_fare])])
    else:
        p.writerow([row[0], "%d" %
                    int(survival_table[1, float(row[1]) - 1, bin_fare])])



test_file.close()
prediction_file.close()



