import numpy as np
from sklearn.metrics import confusion_matrix

def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]

def average_over_positive_values(a):
    a = np.nan_to_num(a, nan=-1)
    positives = a[a != -1]
    avgs = np.average(positives)
    stds = np.std(positives)
    avgs = np.nan_to_num(avgs, nan=-1)
    stds = np.nan_to_num(stds, nan=-1)
    return avgs, stds


def average_over_positive_values_of_2d_array(a):
    a = np.nan_to_num(a, nan=-1)
    positives = [a[a[:, 0] != -1, 0], a[a[:, 1] != -1, 1]]
    avgs = np.average(positives, axis=1)
    stds = np.std(positives, axis=1)
    avgs = np.nan_to_num(avgs, nan=-1)
    stds = np.nan_to_num(stds, nan=-1)
    return avgs, stds


def average_shift(a, index):
    temp = np.copy(a)
    # If the value is nan (-1) in first column, replace it with the value in the next number of ensemble. This makes the confidence shift zero when one of them is not defined.
    temp[0, temp[0] == -1] = temp[1, temp[0] == -1]
    # If the value is nan (-1), replace it with the value in the previous number of ensemble. This makes the confidence shift zero when one of them is not defined.
    if index > 0 :
        temp[index, temp[index] == -1] = temp[index - 1, temp[index] == -1]
    return np.average(temp[index] - temp[0])


    a = np.nan_to_num(a, nan=-1)
    positives = [a[a[:, 0] != -1, 0], a[a[:, 1] != -1, 1]]
    avgs = np.average(positives, axis=1)
    stds = np.std(positives, axis=1)
    avgs = np.nan_to_num(avgs, nan=-1)
    stds = np.nan_to_num(stds, nan=-1)
    return avgs, stds

def average_of_gradient_metrics(a):
    avgs = np.zeros(a.shape[1])
    stds = np.zeros(a.shape[1])
    for i in range(a.shape[1]):
        positives = a[a[:, i] != -1, i]
        avgs[i] = np.average(positives)
        stds[i] = np.std(positives)
    return avgs, stds


def average_of_gradient_metrics_of_2d_array(a):
    avgs = np.zeros((a.shape[1], 2))
    stds = np.zeros((a.shape[1], 2))
    for i in range(a.shape[1]):
        positives = [a[a[:, i, 0] != -1, i, 0], a[a[:, i, 1] != -1, i, 1]]
        avgs[i] = np.average(positives, axis=1)
        stds[i] = np.std(positives, axis=1)
    return avgs, stds


def wigthed_average(value, count):
    return np.sum(value[value != -1] * count[value != -1]) / np.sum(count[value != -1])


def wigthed_average_for_gradient_metrics(value, count):
    avgs = np.zeros(7)
    for i in range(value.shape[1]):
        avgs[i] = np.sum(value[value[:, i] != -1, i] * count[value[:, i] != -1]) / np.sum(count[value[:, i] != -1])
    return avgs


def average_over_gradient_metrics(a):
    avgs = np.average(a, axis=0)
    stds = np.std(a, axis=0)
    return avgs, stds


def wigthed_average_over_gradient_metrics(value, count):
    avgs = np.zeros(value.shape[1])
    for i in range(value.shape[1]):
        metric = value[:, i]
        avgs[i] = np.sum(metric[metric != -1] * count[metric != -1]) / np.sum(count[metric != -1])
    return avgs


def false_alarm_rate(y_true, y_pred):
    TP, TN, FP, FN = classification_scores(y_true, y_pred)
    if FP + TN == 0:
        return -1
    else:
        return FP / (FP + TN)


def classification_scores(y_true, y_pred):
    CM = confusion_matrix(y_true, y_pred)
    if CM.shape[0] <= 1:
        return (0, 0, 0, 0)

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    return (TP, TN, FP, FN)
