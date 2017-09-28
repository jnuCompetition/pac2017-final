#--coding=utf-8--
import pandas as pd
import numpy as np
#from gensim.models import word2vec
import pickle
#import jieba
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties  
font = FontProperties(fname=r"fonts/msyh.ttf", size=10)  
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
def mergeData():
    dataName = 'data.csv'
    dataDir = 'data/'
    files_1= ['第一批训练数据-1（13424） .xls','第一批训练数据-2（9824） .xls',
		  '第一批训练数据-3（2338）.xls','第一批训练数据-4（8696）.xls',
		  '第一批训练数据-5（12965）.xls','第一批训练数据-6（22043）.xls']
    files_2 = ['第二批训练数据-10（266）.xls','第二批训练数据-11（403）.xls',
		  '第二批训练数据-12（510）.xls','第二批训练数据-13（1158）.xls',
		  '第二批训练数据-14（1178）.xls','第二批训练数据-15（4032）.xls',
		  '第二批训练数据-16（8981）.xls','第二批训练数据-1（11）.xls',
		  '第二批训练数据-3（36）.xls','第二批训练数据-4（49）.xls',
		  '第二批训练数据-5（67）.xls','第二批训练数据-6（98）.xls',
		  '第二批训练数据-7（179）.xls','第二批训练数据-8（221）.xls','第二批训练数据-9（253）.xls']
    columns=['cmt','act','fav','eval','penv','pser','adv','time','ptotal','senv','qua','sser','stotal','link']
    data=pd.DataFrame()
    files = [files_1,files_2]
    for i in range(len(files)):
        for fname in files[i]:
            tdata = pd.read_excel(dataDir+fname,header=None)
            tdata = tdata[2:]
            if i == 0:
                tdata.loc[:,13]=None
            data = data.append(tdata,ignore_index=True)
    data.columns=columns
    # Delete dirty data
    data=data[data.act != "银联标签"]
    data.to_csv(dataName,index=False,encoding='utf-8')

def c2e(labels,x):
    if x != x:
        pass
    else:
        return labels.index(x)+1


def getTrain(data,pname,label_name,label_value,subsample_num=1000):
    train = data[data.ix[:,'act']==pname]
    train = train.reset_index(drop=True)
    train.loc[:,label_name] = list(map(lambda x:c2e(label_value,x),train[label_name]))
    
    # Balance the data
    train = imbalance(train,label_name,subsample_num)
    
    res=train.loc[:,['cmt',label_name]]
    _res = np.array(res)
    _data = []
    for elem in _res:
        _data.append((elem[0],elem[1]))
    return _data

def getTrainData(data,pname,label_name,label_value,subsample_num):
    train = data[data.ix[:,'act'] != "其他"]
    train = train.reset_index(drop=True)
    train.loc[:,label_name] = list(map(lambda x:c2e(label_value,x),train[label_name]))
    
    # Balance the data
    train = imbalanceActs(train,label_name,subsample_num)
    
    res=train.loc[:,['cmt',label_name]]
    _res = np.array(res)
    _data = []
    for elem in _res:
        _data.append((elem[0],elem[1]))
    return _data

def getStopWords(stopwords_path):
    stopwords = []
    f = open(stopwords_path,"r")
    while True:
        line = f.readline()
        if len(line) == 0:
            break
        line = "".join(line).strip("\n")
        stopwords.append(line)
    f.close()
    return stopwords

def filterCmt(cutwords,stopwords):
	return [word for word in cutwords if word not in stopwords]

def imbalance(data,label_name,subsample_num):
    subsample_min = 5
    df1 = data[data[label_name] == 2].sample(subsample_num)
    
    num0 = data[data[label_name] == 1].shape[0]
    add_0_num = subsample_num - num0
    df0 = data[data[label_name] == 1]
    for i in range(int(add_0_num/subsample_min)):
        _df0 = data[data[label_name] == 1].sample(subsample_min)
        df0 = pd.concat([df0,_df0])
    
    num2 = data[data[label_name] == 3].shape[0]
    add_2_num = subsample_num - num2
    df2 = data[data[label_name] == 3]
    for i in range(int(add_2_num/subsample_min)):
        _df2 = data[data[label_name] == 3].sample(subsample_min)
        df2 = pd.concat([df2,_df2])
    
    data = pd.concat([df0,df1,df2])
    data = data.reset_index(drop=True)
    return data

def get_w2v(data, vec_len=50):
    w2v = {}
    all_cmts = []
    counter = 0
    for v in data:
        #print (counter)
        #print (v)
        if v[0] != v[0]:
            pass
        else:
            all_cmts.append(list(jieba.cut(v[0].replace('\n',''))))
        counter = counter + 1
    # vec_len = 50
    model = word2vec.Word2Vec(all_cmts,min_count=5,size=vec_len,negative=0,workers=4)
    for key in model.wv.vocab:
        w2v[key]=model[key].tolist()
    w2v['##']=[0]*vec_len
    return w2v

def dumpW2V(data,filePath):
    w2v=get_w2v(getAllCmts(data,"noise"))
    #pickle.dump(w2v,open(filePath,"wb"))
    test=[]
    label=[]
    for key in w2v.keys():
        test.append(key)
    for la in w2v.values():
        label.append(la)
    np.savez('word2vec.npz',cmt=test,label=label)



def loadW2V(filePath):
    return pickle.load(open(filePath,"rb"))
    
def noiseLabel(sample):
    if sample == "其他":
        return 1
    else:
        return 0

def getTrain_(data,label_name,trainNum):
    data.loc[:,"noise"] = list(map(lambda x: noiseLabel(x),data["act"]))
    print(label_name)
    # Balance the data
    train,test = _imbalance(data,label_name)
    
    # Test: choose subample to train data
    train = train.sample(trainNum)
    
    res=train.loc[:,['cmt',label_name]]
    _res = np.array(res)
    _data = []
    for elem in _res:
        _data.append((elem[0],elem[1]))
    
    tres=test.loc[:,['cmt',label_name]]
    _tres = np.array(tres)
    _tdata = []
    for elem in _tres:
        _tdata.append((elem[0],elem[1]))
    
    return _data,_tdata

def getAllCmts(data,label_name):
    res=data.loc[:,['0','2']]
    _res = np.array(res)
    _data = []
    for elem in _res:
        _data.append((elem[0],elem[1]))
    return _data

def _imbalance(data,label_name):
    """
        desc: 'act' == '其他'(label = 1),else (label = 0)
    """
    subsample_num = 19000
    df1 = data[data[label_name] == 1].sample(subsample_num)
    test1 = df1.sample(int(df1.shape[0]*0.2)) 
    df1 = df1.drop(test1.index,axis=0) 
    
    df0 = data[data[label_name] == 0]
    test2 = df0.sample(int(df0.shape[0]*0.2)) 
    df0 = df0.drop(test2.index,axis=0)
       
    data = pd.concat([df0,df1])
    data = data.reset_index(drop=True)

    test = pd.concat([test1,test2])
    test = test.reset_index(drop=True)
    return data,test

def imbalanceActs(data,label_name,subsample_num):
    """
        desc: 'act' == '其他'(label = 1),else (label = 0)
    """
    subsample_min = 100
    df1 = data[data[label_name] == 1].sample(subsample_num)
    df2 = data[data[label_name] == 2].sample(subsample_num)
    df4 = data[data[label_name] == 4].sample(subsample_num)
    
    data_0 = data[data[label_name] == 3]
    num0 = data_0.shape[0]
    add_0_num = subsample_num - num0
    df0 = data_0
    for i in range(int(add_0_num/subsample_min)):
        _df0 = data_0.sample(subsample_min)
        df0 = pd.concat([df0,_df0])
    
    data = pd.concat([df0,df1,df2,df4])
    data = data.reset_index(drop=True)
    return data

def ruleCmp(data):
    posVals = {}
    acts = ["银联62","ApplePay","银联钱包","云闪付"]
    attrs = ["2","3","5","6","7","8"]
    attrName = ["优惠力度","应用评价","服务评价","活动宣传","活动时间","整体评价"]
    for i in range(len(acts)):
        vals = []
        for j in range(len(attrs)):
            act = data[data["1"] == acts[i]]
            totalNum = act.shape[0]
            _act = act[attrs[j]].value_counts()
            if attrs[j] == "6" or attrs[j] == "7":
                val = float(_act["充分"]) / totalNum
            else:
                val = float( _act["好"] )/totalNum
            vals.append(val)
        posVals[acts[i]]=vals
    
    posDf=pd.DataFrame(posVals,index=attrs)
    #plt.figure(figsize=(20,20))
    posDf.plot(kind="bar")
    plt.xticks((0,1,2,3,4,5),tuple(attrName),fontproperties=font,rotation=2)
    plt.title("相同属性不同银联产品积极百分比对比",fontproperties=font)
    plt.xlabel("银联产品属性",fontproperties=font)
    plt.ylabel("积极百分比值",fontproperties=font)
    plt.legend(loc="upper right",prop=font)
    plt.savefig("pos.png")
    plt.close()

    for act in acts: 
        #plt.figure(figsize=(10,10))
        posDf[act].plot(kind="bar",color="green")
        plt.xticks((0,1,2,3,4,5),tuple(attrName),fontproperties=font,rotation=1)
        plt.title(act+"不同属性积极百分比",fontproperties=font)
        plt.xlabel("产品属性",fontproperties=font)
        plt.ylabel("积极百分比值",fontproperties=font)
        plt.savefig(act+".png")
        plt.close()
    return posDf

def to_num(label):
    if label == "好" or label == "充分":
        return 0
    elif label == "中" or label == "中等":
        return 1
    else:
        return 2
def corr(data):
    acts = ["银联62","ApplePay","银联钱包","云闪付"]
    #attrs = ["fav","eval","pser","adv","time","ptotal"]
    attrs = ["2","3","5","6","7","8"]
    attrName = ["优惠力度","应用评价","服务评价","活动宣传","活动时间"]
    for attr in attrs:
        data.loc[:,attr] = list(map(lambda x:to_num(x),data[attr]))
    corrVals = {}
    for act in acts:
        corrs = []
        _data = data[data["1"] == act]
        for i in range(len(attrs)-1):
            corrs.append(_data[attrs[i]].corr(_data["8"]))
        corrVals[act] = corrs
    
    corrDf=pd.DataFrame(corrVals,index=["2","3","5","6","7"])
    #plt.figure(figsize=(50,50))
    corrDf.plot(kind="bar")
    plt.xticks((0,1,2,3,4),tuple(attrName),fontproperties=font,rotation=1)
    plt.title("相同属性不同银联产品相关系数对比",fontproperties=font)
    plt.xlabel("银联产品属性",fontproperties=font)
    plt.ylabel("相关系数",fontproperties=font)
    plt.legend(loc="upper right",prop=font)
    plt.savefig("corr.png")
    plt.close()

   # for act in acts: 
   #     plt.figure(figsize=(10,10))
   #     corrDf[act].plot(kind="bar")
   #     plt.xticks((0,1,2,3,4),tuple(attrName),fontproperties=font)
   #     plt.title(act+"不同属性相关系数",fontproperties=font)
   #     plt.xlabel("产品属性",fontproperties=font)
   #     plt.ylabel("相关系数",fontproperties=font)
   #     plt.savefig("corr/"+act+".jpg")
   #     plt.close()
    return corrVals

def showImbalance(data):
    seq_len = {}
    train = data[data.ix[:,'act']!='其他']
    data1 = train.cmt
    for i in range(len(data1)):
        if data1[i] != data1[i]:
            continue
        words = list(jieba.cut(data1[i].replace('\n',' ')))
        if seq_len.get(len(words)):
            seq_len[len(words)] += 1
        else:
            seq_len[len(words)] = 1
    #print(seq_len)
    list_key = [key for key in seq_len.keys()]
    list_value = [value for value in seq_len.values()]
    plt.plot(list_key, list_value, 'r')
    plt.xlabel('sequence len')
    plt.ylabel('number')
    plt.xlim(0,200)
    plt.ylim(0, 3000)
    plt.title('sequence_len')
    #plt.legend()
    plt.savefig('sequence_len_1.jpg')
    #plt.show()


if __name__ == '__main__':
    
    #data = getStopWords("stopwords")
    #cutWords = "张海鹏把这个处理一下呗！"
    #_cutWords = list(jieba.cut(cutWords.replace('\n','')))
    #_data = filterCmt(_cutWords,data)
    #print (_data)
    #mergeData()
    
    raw_data = pd.read_csv('../_train.csv',low_memory=False,encoding='utf-8')
    #dumpW2V(raw_data,"word2vec.txt")
    # vec = loadW2V("word2vec.pkl")
    # [x,y]=getTrain_(raw_data,"noise",100)
    #actNames = ["银联钱包","银联62","ApplePay","云闪付"]
    #columns=['fav','eval','pser','ptotal']
    #for actName in actNames:
    #    for column in columns:
    #showImbalance(raw_data)
    
    posDf = ruleCmp(raw_data)
    #data = corr(raw_data)
    #data = getTrain(raw_data,'ApplePay','ptotal',[u'差',u'中',u'好'])
