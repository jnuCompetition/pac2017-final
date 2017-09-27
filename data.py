#coding=utf-8

"""
    Desc:   read raw data and make preprocessing, filter outlier
    Author: zhpmatrix
"""

import os
import pickle
import jieba
import pandas as pd
import numpy as np
from gensim.models import word2vec

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
    # Delete NaN
    data = data.dropna(axis=0, how='any')
    data = data.reset_index(drop=True)
    saveData(data, path='../train.csv')
    return data

def get_stop_words(filepath):
    stopwords = []
    f = open(filepath,"r")
    while True:
        line = f.readline()
        if len(line) == 0:
            break
        line = "".join(line).strip("\n")
        stopwords.append(line)
    f.close()
    return stopwords

def filter_comments(cutwords,stopwords):
    return [word for word in cutwords if word not in stopwords]

def dump_data(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_data(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def dump_word2vec(data,vec_len,filepath):
    _word2vec = {}
    corpus = []
    raw_corpus = list( data.ix[:,0] )
    for cmt in raw_corpus:
        corpus.append(list(jieba.cut(cmt.replace('\n',''))))
    model = word2vec.Word2Vec(corpus, min_count=5, size=vec_len, workers=4)
    for key in model.wv.vocab:
        _word2vec[key]=model[key].tolist()
    _word2vec['##']=[0]*vec_len
    with open(filepath, "wb") as f:
        pickle.dump(_word2vec, f)

def load_word2vec(filepath):
    with open(filepath, 'rb') as f:
        _word2vec = pickle.load(f)
    return _word2vec

def c2e(label):
    labels = ['好', '中', '差']
    return labels.index(label) + 1

def get_test(data, activity, _imbalance=False):
    # Don't sampling which make test, so make sample_num equals 0
    return get_train(data, activity, _imbalance, sample_num=0)

def get_train(data, activity, _imbalance, sample_num):
    train = data[data.ix[:, 1]==activity]
    train = train.reset_index(drop=True)
    train.ix[:, 2] = list(map(lambda label:c2e(label),train.ix[:, 2]))
    # Balance the data
    if _imbalance == True:
        #train = imbalance_0(train, activity, sample_num)
        train = imbalance_4(train, activity)
        print(train.ix[:,2].value_counts())
    _res=np.array( train.ix[:,[0, 2]] )
    _data = []
    for elem in _res:
        _data.append((elem[0],elem[1]))
    return _data

def imbalance_0(data, activity, sample_num):
    """
        Desc: sampling data from 3 classes according to the specific sample               number, make the ratio of 3 class equals 1:1:1
    """
    data = data[ data.ix[:,1] == activity]
    series = data.ix[:, 2].value_counts()
    mvalue = int( np.median(series) )
    mindex = series[ series == mvalue].index[0]
    
    svalue = int( np.min(series) )
    sindex = series[ series == svalue].index[0]

    lvalue = int( np.max(series) )
    lindex = series[ series == lvalue].index[0]

    ldf = data[data.ix[:,2] == lindex].sample(sample_num)
    
    mdf = data[data.ix[:,2] == mindex] # 好
    subsample_min = 20
    add_sample_num = sample_num - mvalue
    for i in range(int(add_sample_num / subsample_min)):
        _mdf = mdf.sample(subsample_min)
        mdf = pd.concat([mdf, _mdf])
    
    # Sampling without replacement, sampling number <= total
    sdf = data[data.ix[:,2] == sindex]
    subsample_min = 20
    add_sample_num = sample_num - svalue
    for i in range(int(add_sample_num / subsample_min)):
        _sdf = sdf.sample(subsample_min)
        sdf = pd.concat([sdf, _sdf])

    data = pd.concat([mdf,ldf,sdf]).sample( frac=1 ).reset_index(drop=True)
    return data

def imbalance_1(data, activity):
    """
        Desc: sampling data from '中'(downsampling) and '差'(upsampling) acc              ording to the number of '好', make the ratio of 3 class equals              1:1:1
    """
    data = data[ data.ix[:,1] == activity]
    series = data.ix[:, 2].value_counts()
    data = data[ data.ix[:,1] == activity]
    series = data.ix[:, 2].value_counts()
    
    mvalue = int( np.median(series) )
    mindex = series[ series == mvalue].index[0]
    
    svalue = int( np.min(series) )
    sindex = series[ series == svalue].index[0]

    lvalue = int( np.max(series) )
    lindex = series[ series == lvalue].index[0]

    ldf = data[data.ix[:,2] == lindex].sample(mvalue)
    mdf = data[data.ix[:,2] == mindex] # '好'
    
    # Sampling without replacement, sampling number <= total
    sdf = data[data.ix[:,2] == sindex]
    subsample_min = 20
    add_sample_num = mvalue - svalue
    for i in range(int(add_sample_num / subsample_min)):
        _sdf = sdf.sample(subsample_min)
        sdf = pd.concat([sdf, _sdf])

    data = pd.concat([mdf,ldf,sdf]).sample( frac=1 ).reset_index(drop=True)
    return data

def imbalance_2(data, activity):
    """
        Desc: sampling data from '中'(downsampling) and '差'(upsampling) acc              ording to the number of '好', make the ratio of 3 class equals              1:1:1
    """
    data = data[ data.ix[:,1] == activity]
    series = data.ix[:, 2].value_counts()
    data = data[ data.ix[:,1] == activity]
    series = data.ix[:, 2].value_counts()
    
    mvalue = int( np.median(series) )
    mindex = series[ series == mvalue].index[0]
    
    svalue = int( np.min(series) )
    sindex = series[ series == svalue].index[0]

    lvalue = int( np.max(series) )
    lindex = series[ series == lvalue].index[0]

    ldf = data[data.ix[:,2] == lindex]
    
    mdf = data[data.ix[:,2] == mindex]
    subsample_min = 20
    add_sample_num = lvalue - mvalue
    for i in range(int(add_sample_num / subsample_min)):
        _mdf = mdf.sample(subsample_min)
        mdf = pd.concat([mdf, _mdf])
    
    sdf = data[data.ix[:,2] == sindex]
    subsample_min = 20
    add_sample_num = lvalue - svalue
    for i in range(int(add_sample_num / subsample_min)):
        _sdf = sdf.sample(subsample_min)
        sdf = pd.concat([sdf, _sdf])
    
    data = pd.concat([mdf,ldf,sdf]).sample( frac=1 ).reset_index(drop=True)
    return data

def imbalance_3(data, activity):
    """
        Desc: sampling data from '好'(upsampling) and '差'(upsampling) acc              ording to the number of '中', make the ratio of 3 class equals              1:2:1('好', '中', '差')
    """
    data = data[ data.ix[:,1] == activity]
    series = data.ix[:, 2].value_counts()
    data = data[ data.ix[:,1] == activity]
    series = data.ix[:, 2].value_counts()
    
    mvalue = int( np.median(series) )
    mindex = series[ series == mvalue].index[0]
    
    svalue = int( np.min(series) )
    sindex = series[ series == svalue].index[0]

    lvalue = int( np.max(series) )
    lindex = series[ series == lvalue].index[0]

    ldf = data[data.ix[:,2] == lindex]
    
    mdf = data[data.ix[:,2] == mindex]
    
    subsample_min = 20
    subsample_max = ldf.shape[0] * 0.5
    
    add_sample_num = subsample_max - mvalue
    for i in range(int(add_sample_num / subsample_min)):
        _mdf = mdf.sample(subsample_min)
        mdf = pd.concat([mdf, _mdf])
    
    sdf = data[data.ix[:,2] == sindex]
    subsample_min = 20
    add_sample_num = subsample_max - svalue
    for i in range(int(add_sample_num / subsample_min)):
        _sdf = sdf.sample(subsample_min)
        sdf = pd.concat([sdf, _sdf])
    
    data = pd.concat([mdf,ldf,sdf]).sample( frac=1 ).reset_index(drop=True)
    return data

def imbalance_4(data, activity):
    """
        Desc: sampling solution from Zhengzhou University
    """
    data = data[ data.ix[:,1] == activity]
    series = data.ix[:, 2].value_counts()
    data = data[ data.ix[:,1] == activity]
    series = data.ix[:, 2].value_counts()
    
    mvalue = int( np.median(series) )
    mindex = series[ series == mvalue].index[0]
    
    svalue = int( np.min(series) )
    sindex = series[ series == svalue].index[0]

    lvalue = int( np.max(series) )
    lindex = series[ series == lvalue].index[0]

    ldf = data[data.ix[:,2] == lindex]
    
    mdf = data[data.ix[:,2] == mindex]
    
    subsample_min = 20
    subsample_max = ldf.shape[0] * 1.
    
    add_sample_num = subsample_max - mvalue * 4.
    for i in range(int(add_sample_num / subsample_min)):
        _mdf = mdf.sample(subsample_min)
        mdf = pd.concat([mdf, _mdf])
    
    sdf = data[data.ix[:,2] == sindex]
    subsample_min = 20
    add_sample_num = subsample_max - svalue * 4.
    for i in range(int(add_sample_num / subsample_min)):
        _sdf = sdf.sample(subsample_min)
        sdf = pd.concat([sdf, _sdf])
    
    data = pd.concat([mdf,ldf,sdf]).sample( frac=1 ).reset_index(drop=True)
    return data

if __name__ == '__main__':

    #fileDir = '../train/'
    #train = readTrain(fileDir)
    #print(train.shape)
    #
    #path = '../test/人工智能组2017决赛-0-验证数据公布3300.xlsx'
    #test = _readTest(path)
    #print(test.shape)
    
    #stopwords = get_stop_words("stopwords.txt") 
    #cutwords = "百战归来再读书!"
    #results = filter_comments(list(jieba.cut(cutwords.replace('\n',''))), stopwords)
    #print(results)
    
    train = pd.read_csv('../data/train.csv', low_memory=False, encoding='utf-8')
    activities = ['ApplePay', '银联钱包', '银联62', '云闪付', '其他']
    
    # Get word dict from all samples
    dump_word2vec(train, vec_len=50, filepath='All.pkl')
    
    # Get word dict from each sample
    for activity in activities:
        _train = train[train.ix[:,1] == activity]
        dump_word2vec(_train, vec_len=50, filepath=activity+'.pkl')
        #_word2vec = load_word2vec(filepath='word2vec.pkl')

    # data = get_train(train, '银联62')
