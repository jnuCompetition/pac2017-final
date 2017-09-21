"""
    Desc:   define metrics for triple class classification
    Author: zhpmatrix

"""
import numpy as np
from sklearn.metrics import confusion_matrix

def eval(y_true, y_pred, labels = ['好', '中', '差']):
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
    accuracy = float('%.2f' % (np.trace(matrix) / np.sum(matrix)))
    trueNum = np.sum(matrix, axis=1)
    predNum = np.sum(matrix, axis=0)
    # Precisions
    precisions = {}
    for i in range(num_labels):
            precisions[labels[i]] = float('%.2f' % (matrix[i][i] / predNum[i]))
    # Recalls
    recalls = {}
    for i in range(num_labels):
            recalls[labels[i]] = float('%.2f' % (matrix[i][i] / trueNum[i]))
    # Results
    results = {'accuracy':accuracy,  'precisions':precisions, 'recalls':recalls}
    return results

if __name__ == '__main__':
    
    y_true = ['差', '中', '好', '好', '中', '差', '差']
    y_pred = ['中', '差', '好', '中', '好', '中', '差']
    results = eval(y_true, y_pred)
    print(results) 
