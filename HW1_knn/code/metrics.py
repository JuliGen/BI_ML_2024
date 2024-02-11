import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for pred, test_y in zip(y_pred, y_true):
        if pred == 1 and test_y == 1:
            tp += 1
        if pred == 1 and test_y == 0:
            fp += 1
        if pred == 0 and test_y == 0:
            tn += 1
        if pred == 0 and test_y == 1:
            fn += 1

    print(f"TP = {tp}, fp = {fp}, tn = {tn}, fn = {fn}")
    accuracy = (tp + tn) / (tp + tn + fn + fp)

    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0

    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0

    try:
        f1 = 2 * ((precision * recall) / (precision + recall))
    except ZeroDivisionError:
        f1 = 0

    print(f"Accuracy= {accuracy}\nPrecision= {precision}\nRecall= {recall}\nf1 = {f1} ")


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    correct = 0
    for pred, y in zip(y_pred, y_true):
        if pred == y:
            correct += 1

    accuracy = correct / len(y_pred)
    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    r_sq = 1 - (np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - y_true.mean())))
    return r_sq


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    return sum((y_true - y_pred)**2) / len(y_true)


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    return sum(abs(y_true - y_pred)) / len(y_true)

    