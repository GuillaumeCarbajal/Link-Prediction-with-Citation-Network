import csv
import pandas as pd
import numpy as np

## Load data
with open("training_set.txt", "r") as f:
    reader = csv.reader(f)
    training_set  = list(reader)

training_set = [element[0].split(" ") for element in training_set]

# train
right = pd.read_csv('graph_features_scaled_train.csv')
del right['Unnamed: 0']
left = pd.read_csv('training_features_6.csv', header = None)
train = pd.concat([left, right], axis=1)

X_train = train.as_matrix()

y_train = [i[2] for i in training_set ]

# test
right = pd.read_csv('graph_features_scaled_test.csv')
del right['Unnamed: 0']
left = pd.read_csv('testing_features_6.csv', header = None)
test = pd.concat([left, right], axis=1)

X_test = test.as_matrix()

################
## Prediction
############

from sklearn.ensemble import ExtraTreesClassifier

clf = ExtraTreesClassifier(max_features=None, min_samples_leaf= 20, n_estimators = 300, n_jobs= 3)

clf.fit(X_train, y_train)
pred = clf.predict(X_test)