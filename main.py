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

def build_model(class_num,params):
    embedding_dim = params["embedding_dim"]
    p = params["p"]
    model = Sequential()
    if model_type.lower() == "lstm":
        model.add(Recurrent()
                  .add(LSTM( embedding_dim, 64, p )))
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

def activity_predict(comments, params):
    
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
                                            params["embedding_dim"]) for w in
                                     tokens_label[0]], tokens_label[1]))
    sample_rdd = vector_rdd.map(
        lambda vectors_label: to_sample(vectors_label[0], vectors_label[1], params["embedding_dim"]))
    # Keep order
    predictions = train_model.predict(sample_rdd)
    y_true = [str(map_groundtruth_label( s.label) ) for s in sample_rdd.collect()]
    y_pred = [str(map_predict_label( s )) for s in predictions.collect()]
    # Case study
    print(y_true)
    print(y_pred)
    print(eval(y_true, y_pred, labels=['0', '1', '2']))
    return y_true, y_pred

def predict(data, models, params):
    model = {}
    for modelname in os.listdir(models):
        model[modelname] = models+modelname
    activities = list(data.ix[:,1].unique())
    
    y_true = []
    y_pred = []
    for i in range( len(activities) ):
        comments = get_test(data, activities[i])   
        for key in model.keys():
            if activities[i] in key:
                params['modelpath'] = model[key]
                true, pred = activity_predict(comments, params)
                y_true.extend(true)
                y_pred.extend(pred)
    # Case study
    print(y_true)
    print(y_pred)
    print(eval(y_true, y_pred, labels=['0', '1', '2']))

def saveFig(train_summary, val_summary, max_epoch, activity):
    train_loss = np.array(train_summary.read_scalar("Loss"))
    val_loss = np.array(val_summary.read_scalar("Loss"))
    
    plt.plot(train_loss[:,0], train_loss[:,1], label="curve_train_loss", color="red")
    
    min_idx = np.argmin(val_loss[:,1])
    min_value = float('%.3f' % val_loss[min_idx][1])
    
    plt.plot(val_loss[:,0], val_loss[:,1], label="curve_val_loss", color='yellow')
    plt.scatter(val_loss[:,0], val_loss[:,1], label="scatter_val_loss", color='green')
    plt.title("Results")
    plt.legend()
    plt.savefig('../imgs/'+str(activity)+'-'+str(max_epoch)+'-results.jpg')
    plt.close()
    results = {'Epoch:': min_idx, 'Loss:': min_value}
    return results

def train(sc,
          batch_size,
          sequence_len, max_words, training_split,max_epoch,params):
    
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
                                                params["embedding_dim"]) for w in
                                         tokens_label[0]], tokens_label[1]))
    sample_rdd = vector_rdd.map(
        lambda vectors_label: to_sample(vectors_label[0], vectors_label[1], params["embedding_dim"]))

    train_rdd, val_rdd = sample_rdd.randomSplit(
        [training_split, 1-training_split])

    optimizer = Optimizer(
        model=build_model(3, params),
        training_rdd=train_rdd,
        criterion=ClassNLLCriterion(),
        end_trigger=MaxEpoch(max_epoch),
        batch_size=batch_size,
        optim_method=Adam())
    optimizer.set_validation(
        batch_size = batch_size,
        val_rdd = val_rdd,
        trigger = EveryEpoch(),
        val_method = [Loss()]
    )

    app_name=dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_summary = TrainSummary(log_dir=params["logDir"],app_name=app_name)
    train_summary.set_summary_trigger("Parameters", SeveralIteration(1))
    optimizer.set_train_summary(train_summary)
    val_summary = ValidationSummary(log_dir=params["logDir"],app_name=app_name)
    optimizer.set_val_summary(val_summary)
    
    train_model = optimizer.optimize()
    train_model.save(params['modelpath'], True)
    results = saveFig(train_summary, val_summary, max_epoch, params['activity'])
    
    print('-'*50 + '\n' + str( results ) + '\n'+'-'*50)
    

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-a", "--action", dest="action", default="train")
    parser.add_option("-b", "--batchSize", dest="batchSize", default="160")
    parser.add_option("-e", "--embedding_dim", dest="embedding_dim", default="50")
    parser.add_option("-m", "--max_epoch", dest="max_epoch", default="0")
    parser.add_option("--mdl",dest="model_type", default="gru")

    parser.add_option("-p", "--p", dest="p", default="0.0")
    
    parser.add_option("--act", dest="activity", default="#")

    (options, args) = parser.parse_args(sys.argv)
    batch_size = int(options.batchSize)
    embedding_dim = int(options.embedding_dim)
    max_epoch = int(options.max_epoch)
    p = float(options.p)
    model_type = options.model_type
    activity = options.activity

    sequence_len = 50
    max_words = 1000
    training_split = 0.8
    
    params = {}
    params["word2vec_path"]         =       "../data/word2vec.pkl"
    params["path_to_word_to_ic"]    =       "../data/word_to_ic.pkl"
    params["path_to_filtered_w2v"]  =       "../data/filtered_w2v.pkl"
    params["path_to_stopwords"]     =       "../data/stopwords.txt"        
    params["train"]                 =       "../data/train.csv"
    params["test"]                  =       "../data/test.csv"
    params["logDir"]                =       "../logs/"
    params["activity"]              =       activity
    params["target_value"]          =       ['好', '中', '差']
    params["embedding_dim"]         =       embedding_dim
    params["p"]                     =       p   
    params['modelpath']             =       '../model/'+str(activity)+'-'+str(max_epoch)+'-model.bigdl'
    
    if options.action == 'train':
        # Initialize env
        sc = SparkContext(appName="sa",conf=create_spark_conf())
        init_engine()
        # Train model
        train(sc, batch_size, sequence_len, max_words, training_split, max_epoch, params)
        sc.stop()
    elif options.action == "predict":
        # Initialize env
        sc = SparkContext(appName="sa",conf=create_spark_conf())
        init_engine()
        # Predict model
        test = pd.read_csv(params["test"])
        comments = get_test(test,params['activity'])        
        activity_predict(comments, params)
        sc.stop()
      
    elif options.action == "allpredict":
        # Predict model
        sc = SparkContext(appName="sa",conf=create_spark_conf())
        init_engine()
        data = pd.read_csv(params["test"])
        modelDir = "../model/"
        predict(data, modelDir, params)
        sc.stop()

       # Another conservative method
       # model = {}
       # for modelname in os.listdir(modelDir):
       #     model[modelname] = modelDir+modelname
       # activities = list(data.ix[:,1].unique())
       # 
       # y_true = []
       # y_pred = []
       # for i in range( len(activities) ):
       #     comments = get_test(data, activities[i])   
       #     for key in model.keys():
       #         if activities[i] in key:
       #             params['modelpath'] = model[key]
       #             sc = SparkContext(appName=activities[i],conf=create_spark_conf())
       #             init_engine()
       #             true, pred = activity_predict(comments, params)
       #             sc.stop() 
       #             y_true.extend(true)
       #             y_pred.extend(pred)
       # # Case study
       # print(y_true)
       # print(y_pred)
       # print(eval(y_true, y_pred, labels=['0', '1', '2']))

