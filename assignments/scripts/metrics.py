def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    TP = sum(prediction[ground_truth == True] == True)
    FP = sum(prediction[ground_truth == False] == True)
    FN = sum(prediction[ground_truth == True] == False)
    precision = TP / (FP + TP)
    recall = TP / (TP + FN)
    accuracy = sum(prediction == ground_truth)/len(ground_truth)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    return sum(prediction == ground_truth) / len(ground_truth)
