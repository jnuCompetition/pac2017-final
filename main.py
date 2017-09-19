import os
import pandas as pd

def saveData(data, path):
    data.to_csv(path, index=False, encoding='utf-8')

def readTest(fileDir):
    data = pd.DataFrame()
    for filename in os.listdir(fileDir):
        _data = pd.read_excel(fileDir+filename, header=None, skiprows=[0])
        data = data.append(_data.ix[:, [1,2,3]], ignore_index=False)
    saveData(data, path='../test.csv')
    return data

def readTrain(fileDir):
    data = pd.DataFrame()
    for filename in os.listdir(fileDir):
        _data = pd.read_excel(fileDir+filename, header=None, skiprows=[0,1])
        data = data.append(_data.ix[:, [0,1,8]], ignore_index=False)
    saveData(data, path='../train.csv')
    return data

if __name__ == '__main__':

    fileDir = '../train/'
    train = readTrain(fileDir)
    print(train.shape)
    
    fileDir = '../test/'
    test = readTest(fileDir)
    print(test.shape)
