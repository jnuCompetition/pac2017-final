from bigdl.nn.layer import *
from bigdl.util.common import *
from bigdl.optim.optimizer import *
from main import *

def _predict(modelpath, test):
    """
        Sketch.idea
    """
    model = {}
    for modelname in os.listdir(modelpath):
        model[modelname] = Model.load(modelpath+modelname) 
    sample_num = test.shape[0]
    for i in range(sample_num):
        activity = test.ix[i][1]
        for key in model.keys():
            if len(activity and key) > 0:
                return model[key].predict(test.ix[i][0])

def predict(data, models, params):
    model = {}
    for modelname in os.listdir(modelpath):
        model[modelname] = modelpath+modelname
    activities = list(data.ix[:,1].unique())
    
    y_true = []
    y_pred = []
    for activity in activities:
        comments = get_test(data, activity)   
        for key in model.keys():
            if len(key and activity) > 0:
                params['modelpath'] = model[key]
            true, pred = activity_predict(comments, params)
            y_true.append(true)
            y_pred.append(pred)
    print(eval(y_true, y_pred, labels=['0', '1', '2']))

if __name__ == '__main__':
    pass
