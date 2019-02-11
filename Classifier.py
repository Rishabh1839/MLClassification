# Using supervised learning and labeled training data
# classification of High Def and Ultra High Def resolutions
from sklearn import tree

# include features of the image by resolution x mega pixels
# using the following resolutions in sequence
# 1280 x 720, 1920 x 1080, 3840 x 2160 and 7680 x 4320
# the mega pixels are multiplying the following resolutions above in the list
# to be able to find it's mega pixels
features = [[720, 0.9], [1080, 2.0], [3840, 8.2], [7680, 33.1]]
# labeled the training data
# labels = ["High Def", "High Def", "Ultra High Def", "Ultra High Def"]
labels = [0, 0, 1, 1]

# create a classifier which is a decision tree
clf = tree.DecisionTreeClassifier()

# Do the actual training here for the ML
# Fit is finding the patterns in data
clf = clf.fit(features, labels)

# place unknown data
# predicting its a High Def with 1280 x 720 resolution with 0.9 mega pixels or higher
print(clf.predict([[780, 1.3]]))

# predicting its a Ultra High Def with 3840 x 2160 resolution with 8.2 mega pixels or higher
print(clf.predict([[4820, 10]]))





