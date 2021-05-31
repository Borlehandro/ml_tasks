import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    """
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    """
    true_positive = np.sum(prediction & ground_truth)
    true_negative = np.sum(~prediction & ~ground_truth)
    false_positive = np.sum(prediction & ~ground_truth)
    false_negative = np.sum(~prediction & ground_truth)

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    f1_metric = 2 * precision * recall / (precision + recall)
    # correct / total
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

    return precision, recall, f1_metric, accuracy


def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    true_cnt = np.sum(prediction == ground_truth)
    full_cnt = prediction.size

    return true_cnt / full_cnt
