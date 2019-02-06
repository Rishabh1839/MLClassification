# Using supervised learning and labeled training data
# classification of minivans and sports cars
from sklearn import tree

# include features of the cars by engine hp x seats
features = [[440, 2], [500, 2], [190, 9], [150, 8]]
# labeled the training data
# labels = ["Sports", "Sports", "MiniVan", "MiniVan"]
labels = [0, 0, 1, 1]

# create a classifier which is a decision tree
clf = tree.DecisionTreeClassifier()

# Do the actual training here for the ML
# Fit is finding the patterns in data
clf = clf.fit(features, labels)

# place unknown data
# predicting its a minivan with 160HP and 7 seats
print(clf.predict([[160, 7]]))

# predicting its a sports car with 600HP and 2 seats
print(clf.predict([[600, 2]]))





