import numpy as np


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