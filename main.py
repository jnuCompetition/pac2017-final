import os
import pandas as pd

def saveData(data, path):
    data.to_csv(path, index=False, encoding='utf-8')
def readData(fileDir):
    data = pd.DataFrame()
    for filename in os.listdir(fileDir):
        _data = pd.read_excel(fileDir+filename, header=None, skiprows=[0,1])
        data = data.append(_data.ix[:, [0,8]], ignore_index=False)
    saveData(data, path='data.csv')
    return data
if __name__ == '__main__':
    fileDir = 'data/'
    data = readData(fileDir)
    print(data.shape)
