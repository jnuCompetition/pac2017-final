#coding=utf-8

import re
import pickle
import time
import itertools
import matplotlib
matplotlib.use('Agg')
import datetime as dt
import matplotlib.pyplot as plt

from data import *
from metric import *
from optparse import OptionParser
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *

def text_to_words(review_text,stopwords):
    words = list(jieba.cut(review_text.replace('\n','')))
    words = filter_comments(words,stopwords)
    return words

def analyze_texts(data_rdd,stopwords):
    def index(w_c_i):
        ((w, c), i) = w_c_i
        return (w, (i + 1, c))
    return data_rdd.flatMap(lambda text_label: text_to_words(text_label[0],stopwords)) \
        .map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b) \
        .sortBy(lambda w_c: - w_c[1]).zipWithIndex() \
        .map(lambda w_c_i: index(w_c_i)).collect()

# pad([1, 2, 3, 4, 5], 0, 6)
def pad(l, fill_value, width):
    if len(l) >= width:
        return l[0: width]
    else:
        l.extend([fill_value] * (width - len(l)))
        return l

def to_vec(token, b_w2v, embedding_dim):
    if token in b_w2v:
        return b_w2v[token]
    else:
        return pad([], 0, embedding_dim)

def to_sample(vectors, label, embedding_dim):
    # flatten nested list
    flatten_features = list(itertools.chain(*vectors))
    features = np.array(flatten_features, dtype='float').reshape(
        [sequence_len, embedding_dim])
    return Sample.from_ndarray(features, np.array(label))

def build_model(class_num):
    model = Sequential()
    if model_type.lower() == "lstm":
        model.add(Recurrent()
                  .add(LSTM(embedding_dim, 64, p)))
        model.add(Select(2, -1))
    elif model_type.lower() == "gru":
        model.add(Recurrent()
                  .add(GRU(embedding_dim, 64, p)))
        model.add(Select(2, -1))
    else:
        raise ValueError('model can only be lstm, or gru')

    model.add(Linear(64, 100))
    model.add(Linear(100, class_num))
    model.add(LogSoftMax())
    return model

def map_predict_label(l):
    return np.array(l).argmax()
def map_groundtruth_label(l):
    return int(l[0] - 1)

def predict(comments,embedding_dim,params):
    
    train_model = Model.load(params['modelpath'])
    word_to_ic = load_data(params['path_to_word_to_ic'])
    filtered_w2v = load_data(params['path_to_filtered_w2v'])
    stopwords = get_stop_words(params["path_to_stopwords"])
    
    data_rdd = sc.parallelize(comments, 2)
    bword_to_ic = sc.broadcast(word_to_ic)
    bfiltered_w2v = sc.broadcast(filtered_w2v)

    tokens_rdd = data_rdd.map(lambda text_label:
                          ([w for w in text_to_words(text_label[0],stopwords) if
                            w in bword_to_ic.value], text_label[1]))
    padded_tokens_rdd = tokens_rdd.map(
        lambda tokens_label: (pad(tokens_label[0], "##", sequence_len), tokens_label[1]))
    vector_rdd = padded_tokens_rdd.map(lambda tokens_label:
                                   ([to_vec(w, bfiltered_w2v.value,
                                            embedding_dim) for w in
                                     tokens_label[0]], tokens_label[1]))
    sample_rdd = vector_rdd.map(
        lambda vectors_label: to_sample(vectors_label[0], vectors_label[1], embedding_dim))
    
    # Keep order
    predictions = train_model.predict(sample_rdd)
    y_true = [str(map_groundtruth_label( s.label) ) for s in sample_rdd.collect()]
    y_pred = [str(map_predict_label( s )) for s in predictions.collect()]
    # Case study
    print(y_true)
    print(y_pred)
    print(eval(y_true, y_pred, labels=['0', '1', '2']))

def saveFig(summary,figParams):
    # Train results
    loss = np.array(summary.read_scalar(figParams["scalar_name"]))
    plt.figure(figsize = (12,12))
    plt.plot(loss[:,0],loss[:,1],label=figParams["title"])
    plt.xlim(0,loss.shape[0]+10)
    plt.title(figParams["title"])
    plt.savefig(figParams["title"]+'.jpg')

def train(sc,
          batch_size,
          sequence_len, max_words, embedding_dim, training_split,params):
    
    print('Processing text dataset')
    raw_data = pd.read_csv(params["train"],low_memory=False,encoding='utf-8')
    texts = get_train(raw_data,activity=params['activity'])
    
    stopwords = get_stop_words(params["path_to_stopwords"])
    data_rdd = sc.parallelize(texts, 2)
    word_to_ic = analyze_texts(data_rdd,stopwords)

    # Only take the top wc between [10, sequence_len]
    word_to_ic = dict(word_to_ic[10: max_words])
    dump_data(word_to_ic,params['path_to_word_to_ic'])
    
    bword_to_ic = sc.broadcast(word_to_ic)

    w2v = load_word2vec(params['word2vec_path'])
    
    filtered_w2v = dict((w, v) for w, v in w2v.items() if w in word_to_ic)
    dump_data(filtered_w2v, params['path_to_filtered_w2v'])
    
    bfiltered_w2v = sc.broadcast(filtered_w2v)

    tokens_rdd = data_rdd.map(lambda text_label:
                              ([w for w in text_to_words(text_label[0],stopwords) if
                                w in bword_to_ic.value], text_label[1]))
    padded_tokens_rdd = tokens_rdd.map(
        lambda tokens_label: (pad(tokens_label[0], "##", sequence_len), tokens_label[1]))
    vector_rdd = padded_tokens_rdd.map(lambda tokens_label:
                                       ([to_vec(w, bfiltered_w2v.value,
                                                embedding_dim) for w in
                                         tokens_label[0]], tokens_label[1]))
    sample_rdd = vector_rdd.map(
        lambda vectors_label: to_sample(vectors_label[0], vectors_label[1], embedding_dim))

    train_rdd, val_rdd = sample_rdd.randomSplit(
        [training_split, 1-training_split])

    optimizer = Optimizer(
        model=build_model(3),
        training_rdd=train_rdd,
        criterion=ClassNLLCriterion(),
        end_trigger=MaxEpoch(max_epoch),
        batch_size=batch_size,
        optim_method=Adam())
    optimizer.set_validation(
        batch_size = batch_size,
        val_rdd = val_rdd,
        trigger = EveryEpoch()
    )

    app_name=dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_summary = TrainSummary(log_dir=params["logDir"],app_name=app_name)
    train_summary.set_summary_trigger("Parameters", SeveralIteration(1))
    optimizer.set_train_summary(train_summary)
    val_summary = ValidationSummary(log_dir=params["logDir"],app_name=app_name)
    optimizer.set_val_summary(val_summary)
    

    train_model = optimizer.optimize()
    train_model.save(params['modelpath'], True)
    
    figParams = {}
    summary = val_summary
    figParams["title"] = "top1"
    figParams["scalar_name"] = "Top1Accuracy"
    saveFig(summary,figParams)
    
    print("Train over!")

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-a", "--action", dest="action", default="train")
    parser.add_option("-b", "--batchSize", dest="batchSize", default="160")
    parser.add_option("-e", "--embedding_dim", dest="embedding_dim", default="50")
    parser.add_option("-m", "--max_epoch", dest="max_epoch", default="2")
    parser.add_option("--model", dest="model_type", default="gru")
    parser.add_option("-p", "--p", dest="p", default="0.1")

    (options, args) = parser.parse_args(sys.argv)
    batch_size = int(options.batchSize)
    embedding_dim = int(options.embedding_dim)
    max_epoch = int(options.max_epoch)
    p = float(options.p)
    model_type = options.model_type
    
    sequence_len = 50
    max_words = 1000
    training_split = 0.8
    
    params = {}
    params['modelpath'] =           '../data/model.bigdl'
    params["word2vec_path"] =       "../data/word2vec.pkl"
    params["path_to_word_to_ic"]=   "../data/word_to_ic.pkl"
    params["path_to_filtered_w2v"]= "../data/filtered_w2v.pkl"
    params["path_to_stopwords"]=    "../data/stopwords.txt"        
    params["train"] =               "../data/train.csv"
    params["test"] =                "../data/test.csv"
    params["logDir"] =              "logs/"
    params["activity"] =            "ApplePay"
    params["target_value"] =        ['好', '中', '差']
    if options.action == 'train':
        # Initialize env
        sc = SparkContext(appName="sa",conf=create_spark_conf())
        init_engine()
        # Train model
        train(sc, batch_size, sequence_len, max_words, embedding_dim, training_split, params)
        sc.stop()
    elif options.action == "predict":
        # Initialize env
        sc = SparkContext(appName="sa",conf=create_spark_conf())
        init_engine()
        # Predict model
        sentences = [("一个严肃的问题，招行双币一卡通，VISA+银联，是否可以用卡号+有效期通过visa消费 借记卡也可以这么消费啊？太不安全了",1),
                ("applepay只能是银联吗？ wlmouse 发表于 2016-2-26 09:57 地区变成美国，能绑外卡不。中国即可绑就像之前地区改美国也能绑银联卡一样",2),
                ("关于目前的部分Pos机对于ApplePay闪付的一个缺陷 505597029 发表于 2016-2-22 14:54 体验写的很仔细，点赞…不过很多收银员都不会用闪>    付才是真的缺陷…",3),
                ("Samsung Pay在中国注定干不过Apple Pay的原因 我看三星根本没想这么多。怎么样它都占据了手机器件，还有半导体芯片的上游，这个行业在他就会一直爽下去。他只要保证时刻不掉队即可。过几年几十年又出另一个苹果或者另几个，总有哪个死了，不过三星怎么都不会死。因为大家都得用它的器件。",3)]
        test = pd.read_csv(params["test"])
        comments = get_test(test,params['activity'])        
        predict(comments, embedding_dim, params)
        sc.stop()
