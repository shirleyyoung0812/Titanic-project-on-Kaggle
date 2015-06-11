__author__ = 'shirleyyoung'

import csv
import numpy as np

test_file = open('/Users/shirleyyoung/Documents/Kaggle/Titantic/test.csv')
test_object = csv.reader(test_file)
header = test_file.__next__()

prediction_file = open('genderBasedModel.csv','w')
prediction_file_object = csv.writer(prediction_file)

prediction_file_object.writerow(["PassengerId", "Survived"])
for row in test_object:
    if row[3] == 'female':
        prediction_file_object.writerow([row[0], '1'])
    else:
        prediction_file_object.writerow([row[0], '0'])

test_file.close()
prediction_file.close()