
# Script that tries to visualize the data and give an idea of what we're working with

import numpy as np
import read
import utils
from sklearn import svm
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt


def PCA(filename):
    """
    Visualize in 2D.
    """
    reader = read.ChunkDictReader(filename)

    # read only 2000 lines
    data = get_chunk(reader, 2000)

    # visualize data using PCA
    utils.PCA_plot(data["X"], data["y"])


def figure_correlations(data):
    """
    Test individual feature correlations and plot them.
    :data: should be a dictionary containing X_train and y_train categories (standard for most methods here).
    """
    correlations = []
    categories = data["categories"]
    ind = range(len(categories))

    for idx in range(len(categories)):
        feature = data["X_train"][:, idx]
        # get the correlation coefficient and append to list for plotting
        correlation = utils.pearson_correlation(feature, data["y_train"])
        correlations.append(np.absolute(correlation))

    plt.bar(ind, correlations, 0.35)
    plt.title("Feature correlations")
    plt.ylabel("Absolute degree of correlation")
    plt.xlabel("Features")
    plt.xticks(ind, categories, rotation="vertical")

    # make sure that labels fit inside plot and show the plot
    plt.tight_layout()
    plt.show()


def analyse(data):
    #################################
    # Plot pearson correlations
    #################################
    figure_correlations(data)
    raw_input("Press enter to continue.")

    #################################
    # Choose best polynomial degree
    #################################
    degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    utils.plot_poly(degrees, data, svm.SVC, log_loss)
    raw_input("Press enter to continue.")

    #################################
    # Choose best C value
    #################################
    c_vals = [13, 10, 3, 2, 1, 0.5, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    utils.choose_C_for_classifier(c_vals, data, svm.SVC, log_loss)
    raw_input("Press enter to continue.")

    #################################
    # Visualize using PCA
    #################################
    utils.PCA_plot(data["X_train"], data["y_train"])
    raw_input("Press enter to continue.")


    #################################
    # Plot a learning curve
    #################################
    # plotting a learning curve to diagnose any problems in overfitting or lack of data
    model = svm.SVC()
    # do not start at one, to ensure that we have enough classes (>=2) for the fit() method to work
    utils.plot_learning_curve(range(40, 517, 10), data, model, log_loss)

    raw_input("Press enter to continue.")

if __name__ == '__main__':
    # initialize data object that will contain training and validation
    data = {}

    reader = read.ChunkDBReader("forestfires.db", "fires")

    ##############################
    # LOAD AND SORT DATA
    ##############################

    # this gets all the training data available
    training_data = reader.next(1000)
    training_data = read.process_for_training(training_data, "label", ["area", "ISI", "RH", "day", "wind"])
    # add training data to data object
    data["X_train"] = training_data["X"]
    data["y_train"] = training_data["y"]
    data["categories"] = training_data["categories"]
    # this gets all the validation data available
    validation_data = reader.next_val(1000)
    validation_data = read.process_for_training(validation_data, "label", ["area", "ISI", "RH", "day", "wind"])
    # add validation data to data object
    data["X_val"] = validation_data["X"]
    data["y_val"] = validation_data["y"]

    ##############################
    # NORMALIZE DATA
    ##############################
    normX, mu, sigma = utils.normalize(data["X_train"])
    data["X_train"] = normX
    data["X_val"] = utils.normalize_with(data["X_val"], mu, sigma)

    #################################
    # Raise data to second degree, which has proved to bear the smallest error for validation
    #################################
    data_mod = {}
    data_mod['X_train'] = utils.map_to_poly(data["X_train"], 2)
    data_mod['X_val'] = utils.map_to_poly(data['X_val'], 2)
    data_mod['y_train'], data_mod['y_val'] = data['y_train'], data['y_val']

    # analyse(data_mod)

    print "Using optimal values to make a model..."
    opt = svm.SVC(C=0.1)
    opt.fit(data_mod["X_train"], data_mod["y_train"])
    print "Training score:"
    print opt.score(data_mod["X_train"], data_mod["y_train"])
    print "Validation score:"
    print opt.score(data_mod["X_val"], data_mod['y_val'])
    print "Naive classifier (dummy) score:"
    print utils.dummy(data_mod)
