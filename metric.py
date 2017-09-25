#coding=utf-8

"""
    Desc:   define metrics for triple class classification
    Author: zhpmatrix

"""
import numpy as np
from sklearn.metrics import confusion_matrix

def eval(y_true, y_pred, labels):
    """
        Desc: eval predictions
        Params:
            y_true -- true label
            y_pred -- predict label
            labels -- labels of y
        Returns:
            results -- dictionary with accuracy, precisions(p), recalls(r), and p, r are for each label 
    """
    
    num_labels = len(labels)
    matrix = confusion_matrix(y_true, y_pred, labels)
    if np.sum(matrix) == 0:
        accuracy = 0.0
    else:
        accuracy = float('%.2f' % (np.trace(matrix) / np.sum(matrix)))
    trueNum = np.sum(matrix, axis=1)
    predNum = np.sum(matrix, axis=0)
    # Precisions
    precisions = {}
    for i in range(num_labels):
        if predNum[i] == 0:
            precisions[labels[i]] = 0.0
        else:
            precisions[labels[i]] = float('%.2f' % (matrix[i][i] / predNum[i]))
    # Recalls
    recalls = {}
    for i in range(num_labels):
        if trueNum[i] == 0:
            recalls[labels[i]] = 0.0
        else:
            recalls[labels[i]] = float('%.2f' % (matrix[i][i] / trueNum[i]))

    # F1-score
    f1_score = {}
    for label in labels:
        precision = precisions[label]
        recall = recalls[label]
        if precision + recall == 0:
            f1_score[label] = 0.0
        else:
            f1_score[label] = float('%.2f' % (2. * (precision * recall)/ (precision + recall)) )
    results = {'accuracy': accuracy, 'f1_score':f1_score, 'precisions':precisions, 'recalls':recalls}
    return results

if __name__ == '__main__':
    y_true = ['0', '2', '1','1']
    y_pred = ['0', '1', '2','0']
    results = eval(y_true, y_pred, labels=['0','1','2'])
    print(results)
