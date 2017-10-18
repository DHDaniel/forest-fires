# Program that trains a Support Vector Machine on the forest fire data to predict the likelihood of a forest fire.
# The methods included here are not adapted for stochastic gradient descent, since the number of items being trained on is relatively small.

import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

import read
import utils

def train_classifier(data, classifier_func):
    """
    Trains a support vector machine using the 'X_train' and 'y_train' attributes in :data:. Returns a model.
    :classifier_func: should be the function to call to create a new sklearn classifier.
    """
    classifier = classifier_func()
    classifier.fit(data['X_train'], data['y_train'])

    return classifier


if __name__ == '__main__':

    # initialize data object that will contain training and validation
    data = {}

    reader = read.ChunkDBReader("forestfires.db", "fires")

    ##############################
    # LOAD AND SORT DATA
    ##############################

    # this gets all the training data available
    training_data = reader.next(1000)
    training_data = read.process_for_training(training_data, "label", ["area"])
    # add training data to data object
    data["X_train"] = training_data["X"]
    data["y_train"] = training_data["y"]
    data["categories"] = training_data["categories"]
    # this gets all the validation data available
    validation_data = reader.next_val(1000)
    validation_data = read.process_for_training(validation_data, "label", ["area"])
    # add validation data to data object
    data["X_val"] = validation_data["X"]
    data["y_val"] = validation_data["y"]

    print data["X_train"][0]
    print data["y_train"][0]
    ##############################
    # NORMALIZE DATA
    ##############################
    normX, mu, sigma = utils.normalize(data["X_train"])
    data["X_train"] = normX
    data["X_val"] = utils.normalize_with(data["X_val"], mu, sigma)

    clf = train_classifier(data, DummyClassifier)
    print "Dummy classifier trained."
    print "Score:", clf.score(data["X_val"], data["y_val"])

    clf = train_classifier(data, svm.SVC)
    print "SVM trained."
    print "Validation score:", clf.score(data["X_val"], data["y_val"])
    print "Training score:", clf.score(data["X_train"], data["y_train"])

    clf = train_classifier(data, LogisticRegression)
    print "Logistic regressor trained."
    print "Score:", clf.score(data["X_val"], data["y_val"])
