"""
    Desc:   read raw data and make preprocessing, filter outlier
    Author: zhpmatrix
"""

import os
import pandas as pd

def saveData(data, path):
    """
    Desc:   save data to csv file with utf-8 encoding

    Params:
        data -- input data
        path -- save data path

    Returns:
        None
    """
    data.to_csv(path, index=False, encoding='utf-8')

def readTest(fileDir):
    """
    Desc:   preprocess raw test data

    Params:
        fileDir -- file dir of raw training data, diff with _readTest
    
    Returns:
        data -- new test data
    """
    data = pd.DataFrame()
    for filename in os.listdir(fileDir):
        _data = pd.read_excel(fileDir+filename, header=None, skiprows=[0])
        data = data.append(_data.ix[:, [1,2,3]], ignore_index=False)
    # Filter outlier
    data.ix[:,2].replace('银联62活动', '银联62', inplace=True)
    data.ix[:,2].replace(['Applepay','Apple Pay'], 'ApplePay', inplace=True)
    # Rename columns
    data.columns = [i for i in range(3)] 
    saveData(data, path='../test.csv')
    return data

def _readTest(path):
    """
    Desc:   preprocess raw test data

    Params:
        path -- file path of raw training data
    
    Returns:
        data -- new test data
    """
    data = pd.read_excel(path, header=None, skiprows=[0])
    data = data.ix[:, [1,2,3]]
    # Filter outlier
    data.ix[:,2].replace('银联62活动', '银联62', inplace=True)
    data.ix[:,2].replace(['Applepay','Apple Pay'], 'ApplePay', inplace=True)
    # Rename columns
    data.columns = [i for i in range(3)] 
    saveData(data, path='../test.csv')
    return data

def readTrain(fileDir):
    """
    Desc:   preprocess raw training data

    Params:
        fileDir -- dir path of raw training data
    
    Returns:
        data -- new training data
    """

    data = pd.DataFrame()
    for filename in os.listdir(fileDir):
        _data = pd.read_excel(fileDir+filename, header=None, skiprows=[0,1])
        data = data.append(_data.ix[:, [0,1,8]], ignore_index=False)
    # Filter outlier
    data = data[data.ix[:, 1] != '银联标签']
    # Rename columns
    data.columns = [i for i in range(3)] 
    saveData(data, path='../train.csv')
    return data

if __name__ == '__main__':

    fileDir = '../train/'
    train = readTrain(fileDir)
    print(train.shape)
    
    path = '../test/人工智能组2017决赛-0-验证数据公布3300.xlsx'
    test = _readTest(path)
    print(test.shape)