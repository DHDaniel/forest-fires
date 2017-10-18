
import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.dummy import DummyClassifier

def rmsle(predicted,real):
    sum=0.0
    for x in range(len(predicted)):
        if (predicted[x] + 1) < 0:
            predicted[x] = 0
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5

def normalize(X):
    """
    Normalizes the data X by subtracting the mean of each feature and dividing by the standard deviation. Returns a normalized matrix X along with the values of mu (mean) and sigma (stddev)
    """

    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    normX = (X - mu) / sigma

    return normX, mu, sigma

def normalize_with(X, mu, sigma):
    """
    Performs normalization on each column of the feature vector X using the mu and sigma provided.
    """
    normX = (X - mu) / sigma
    return normX

def choose_C_for_classifier(cs, data, model_func, loss):
    """
    Tries a LogisticRegression model or an SVM model with different values for C, the inverse regularization parameter. Does not support stochastic gradient descent regressors.
    Plots a graph of c vs. loss.
    :cs: - list of floats that represent values for C.
    :
    """
    print "Finding best c value..."

    n = len(cs)

    cval_acc = []
    train_acc = []
    for c in cs:
        print "Training model c", c
        # create new model to test c values
        model_test = model_func(C=c)

        model_test.fit(data["X_train"], data["y_train"])

        predict_train = model_test.predict(data["X_train"])
        predict_val = model_test.predict(data["X_val"])

        #print predict_train, data["y_train"]

        cval_acc.append(loss(data["y_val"], predict_val))
        train_acc.append(loss(data["y_train"], predict_train))

    plt.plot(range(n), cval_acc, "r-", label="Cross validation loss")
    plt.plot(range(n), train_acc, "b-", label="Training loss")
    plt.title("Errors for different c values")
    plt.legend()
    plt.xlabel("c value")
    plt.ylabel("Model loss")
    # make sure that the x-values are plotted with equal spacing and proper labels
    plt.xticks(range(n), cs)
    plt.show()


def plot_learning_curve(rng, data, model, loss):
    """
    Plots a learning curve based on the model provided.

    rng: Range of values that the learning curve will be plotted over (value of number of training examples)
    data: a dictionary containing values for the keys "X_train", "y_train", "X_val" and "y_val"
    model: A model with the "fit" method and "predict" (sklearn)
    loss: function to calculate the cost. Accepts values for predictions and ground-truth labels.
    """
    # plotting learning curves to diagnose model
    print "Generating learning curve..."
    err_train = []
    err_val = []
    for i in rng:
        print "Step", i
        train_subset = data["X_train"][0:i, :]
        ans_subset = data["y_train"][0:i]
        # train on subset of examples
        model.fit(train_subset, ans_subset)
        test_pred = model.predict(train_subset)
        cval_pred = model.predict(data["X_val"])

        err_train.append(loss(ans_subset, test_pred))
        err_val.append(loss(data["y_val"], cval_pred))

    plt.plot(rng, err_train, "r-", label="Training error")
    plt.plot(rng, err_val, "b-", label="Cross validation error")
    plt.legend()
    plt.ylabel("Error")
    plt.xlabel("Number of training examples")
    plt.title("Learning curve")
    plt.show()


def PCA_plot(X, y):
    """
    Performs Principal Component Analysis on the dataset, and plots the reduced features of multidimensional X against y.
    """
    X, mu, sigma = normalize(X)
    y, mu_y, sigma_y = normalize(y)

    # dimesions of dataset
    m, n = X.shape

    # computing covariance matrix
    covariance = (1 / float(m)) * np.dot(np.transpose(X), X)

    # perfoming singular vector decomposition to get planes
    U, S, V = svd(covariance, compute_uv=True)

    # we want to reduce to one dimension, so only grab the first eigenvector
    U_reduce = U[:, 0:1]

    # Z is the reduced dimension
    Z = np.dot(X, U_reduce)

    print "Visualizing data using PCA..."

    plt.scatter(Z, y, c='r', marker='x')
    plt.title("PCA Reduction")
    plt.xlabel("Features")
    plt.ylabel("Y values")
    plt.show()


def map_to_poly(X, degree):
    """
    Maps all the features of X to polynomial features of degree (degree)
    """
    # initialize
    X_poly = np.zeros((X.shape[0], 0))

    for idx in range(X.shape[1]):
        mapped = np.reshape(X[:, idx], (X.shape[0], 1))
        for num in range(1, degree + 1):
            new_col = mapped ** num
            X_poly = np.append(X_poly, new_col, axis=1)

    return X_poly


def plot_poly(degrees, data, model_func, loss):
    """
    Maps the values in data (X_train and X_val) onto all the degrees in the list (degrees), and then plots a curve of the error using (model) and (loss)
    """
    err_train = []
    err_val = []

    for deg in degrees:
        X_train = map_to_poly(data["X_train"], deg)
        X_val = map_to_poly(data["X_val"], deg)

        model = model_func()
        model.fit(X_train, data["y_train"])

        # predict values using fitted model
        pred_t = model.predict(X_train)
        pred_v = model.predict(X_val)

        # add corresponding errors using provided loss function
        err_train.append(loss(data["y_train"], pred_t))
        err_val.append(loss(data["y_val"], pred_v))

    plt.plot(degrees, err_train, "r-", label="Training error")
    plt.plot(degrees, err_val, "b-", label="Validation error")
    plt.legend()
    plt.xlabel("Degree")
    plt.ylabel("Loss")
    plt.title("Degree of polynomial vs loss")
    plt.show()

def pearson_correlation(X, Y):
    """
    Calculates the pearson correlation coefficient between elements X and Y (two one-dimensional vectors). It is a measure of how correlated the two variables are.
    Values close to 1 or -1 indicate strong positive/negative correlation. Values closer to 0 indicate a smaller degree of correlation.
    """
    # get the covariance cov(X, Y)
    covariance = np.cov(X, Y, bias=True)[0][1]
    r = covariance / float(np.std(X) * np.std(Y))

    return r


def dummy(data):
    """
    Trains a dummy classifier on the data and evaluates it. Returns the score of the classifier.
    """
    clf = DummyClassifier()
    clf.fit(data["X_train"], data["y_train"])
    return clf.score(data["X_val"], data["y_val"])
