#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

### read in data dictionary, convert to numpy array
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
features = ["salary", "bonus"]

data = featureFormat(data_dict, features)

### your code below

sal = []
for point in data:
    salary = point[0]
    bonus = point[1]
    sal.append(salary)
    # matplotlib.pyplot.scatter(salary, bonus)

sal = sorted(sal)
print sal
lastThreeMaxSal = sal[-3:]
print lastThreeMaxSal

for key, value in data_dict.items():
    if value['salary'] in lastThreeMaxSal:
        print (key)
        data_dict.pop(key)

data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter(salary, bonus)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
