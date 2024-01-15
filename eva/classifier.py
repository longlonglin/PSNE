import argparse
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.io import loadmat
from sklearn.utils import shuffle as skshuffle
from gensim.models import Word2Vec, KeyedVectors
from collections import defaultdict
from scipy import sparse
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
import sys
import matplotlib.pyplot as plt

exis =list()
class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = sparse.lil_matrix(probs.shape)

        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            for label in labels:
                all_labels[i, label] = 1
        return all_labels


def load_embeddings(embeddings_file):
    # load embeddings from word2vec format file
    model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
    features_matrix = np.asarray([model[str(node)] for node in range(len(model.index2word))])
    return features_matrix

def load_embeddings4youtube(embeddings_file):
    # load embeddings from word2vec format file
    model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
    features_matrix = np.asarray([model[str(node)] for node in exis])
    return features_matrix

def load_labels(labels_file, nodesize):
    # load label from label file, which each line i contains all node who have label i
    with open(labels_file) as f:
        context = f.readlines()
        print('class number: ', len(context))
        label = sparse.lil_matrix((nodesize, len(context)))

        for i, line in enumerate(context):
            line = map(int, line.strip().split('\t'))
            for node in line:
                label[node, i] = 1

    print('load done')
    return label


def load_labels_flickr(labels_file, nodesize):
    # load label from label file, which each line i contains all node who have label i
    with open(labels_file) as f:
        context = f.readlines()
        print('class number: ', 195)
        label = sparse.lil_matrix((nodesize, 195))

        for i, line in enumerate(context):
            # print(type(line),line)python classifier.py
            one_line = [int(x) for x in line.split(',')]
            # print(type(one_line), one_line)
            node_ = one_line.pop(0)
            for label_ in one_line:
                label[node_-1 , label_-1] = 1

    print('load done')
    return label

# def load_labels_youtube(labels_file, nodesize):
#     # load label from label file, which each line i contains all node who have label i
#     with open(labels_file) as f:
#         context = f.readlines()
#         print('class number: ', 47)
#         label = sparse.lil_matrix((nodesize, 47))

#         for i, line in enumerate(context):
#             # print(type(line),line)python classifier.py
#             one_line = [int(x) for x in line.split(',')]
#             # print(type(one_line), one_line)
#             node_ = one_line.pop(0)
#             for label_ in one_line:
#                 label[node_-1 , label_-1] = 1

#     print('load done')
#     return label


def load_labels_blog(labels_file, nodesize):
    # load label from label file, which each line i contains all node who have label i
    with open(labels_file) as f:
        context = f.readlines()
        print('class number: ', 39)
        label = sparse.lil_matrix((nodesize, 39))

        for i, line in enumerate(context):
            # print(type(line),line)python classifier.py
            one_line = [int(x) for x in line.split(',')]
            # print(type(one_line), one_line)
            node_ = one_line.pop(0)
            for label_ in one_line:
                label[node_-1 , label_-1] = 1

    print('load done')
    return label


def _load_node2vecPPI_label(labels_file):
    # load label from label file, which each line i contains all node who have label i
    with open(labels_file) as f:
        context = f.readlines()
        print('node number: ', len(context))
        label = sparse.lil_matrix((len(context), 50))

        for i, line in enumerate(context):
            one_line = [int(x) for x in line.split()]
            node_ = one_line.pop(0)
            for label_ in one_line:
                label[node_ - 1, label_ - 1] = 1
    print('load done')

    return label
def load_labels_orkut(labels_file):
    # load label from label file, which each line i contains all node who have label i
    with open(labels_file) as f:
        context = f.readlines()
        print('node number: ', len(context))
        label = sparse.lil_matrix((len(context), 50))

        for i, line in enumerate(context):
            one_line = [int(x) for x in line.split()]
            node_ = one_line.pop(0)
            for label_ in one_line:
                label[node_, label_] = 1
    print('load done')

    return label
def _get_label_node_embedding_youtube(labels_file):
    print('start fliter nodes')

    lable4exis ={}
    with open(labels_file) as f:
         
         context = f.readlines()
         for i,line in enumerate(context):
             one_line = [int(x) for x in line.split(',')]
             if one_line[0]-1 not in exis:
                exis.append(one_line[0]-1)
         for k,value in enumerate(exis):
             lable4exis.update({value:k})
         print('the number of label vertices',len(exis))  
         new_labels_matrix =  sparse.lil_matrix((len(exis), 47))
         
         for i,line in enumerate(context):
                one_line = [int(x) for x in line.split(',')]
                row_index = lable4exis[one_line[0]-1]
                new_labels_matrix[row_index,one_line[1]-1]=1  
         
    return  new_labels_matrix


def _get_label_node_embedding_orkut(labels_file):
    print('start fliter nodes for orkut')

    
    with open(labels_file) as f:
         context = f.readlines()
         for i,line in enumerate(context):
             one_line = [int(x) for x in line.split(' ')]
             if one_line[0] not in exis:
                exis.append(one_line[0])
        #  for k,value in enumerate(exis):
        #      lable4exis.update({value:k})
         print('the number of label vertices',len(exis))  
         new_labels_matrix =  sparse.lil_matrix((len(exis), 100))
         for i,line in enumerate(context):
                one_line = [int(x) for x in line.split(' ')]
                one_line.pop(0)
                for l in one_line: 
                    new_labels_matrix[i,l]=1  
         
    return  new_labels_matrix

def evaluate(label, emb, shuffle,graph_name):
    plt.close('all')
    if graph_name!='youtube' and graph_name!='orkut' :
        print('not youtube...')
        features_matrix = load_embeddings(emb)
        nodesize = features_matrix.shape[0]
    if graph_name=='others':
        label_matrix = load_labels(label, nodesize)
        # label_matrix = _load_node2vecPPI_label(label)
    if graph_name=='flickr':
        label_matrix = load_labels_flickr(label, nodesize)
    if graph_name=='youtube':
        print('is youtube...')
        label_matrix = _get_label_node_embedding_youtube(label)
        features_matrix = load_embeddings4youtube(emb)
        nodesize = features_matrix.shape[0]
    if graph_name=='blog':
        label_matrix = load_labels_blog(label, nodesize)
    if graph_name=='orkut':
        print('is orkut...')
        label_matrix = _get_label_node_embedding_orkut(label)
        features_matrix = load_embeddings4youtube(emb)
        
    number_shuffles = shuffle
    print(features_matrix.shape)
    print(label_matrix.shape)
        
   
   
       
    shuffles = []
    for x in range(number_shuffles):
        shuffles.append(skshuffle(features_matrix, label_matrix))

    all_results = defaultdict(list)

    training_percents = [0.1,0.3,0.5,0.7,0.9]
    
    for train_percent in training_percents:
        for shuf in shuffles:
            X, y = shuf
            training_size = int(train_percent * nodesize)

            X_train = X[:training_size, :]
            y_train = y[:training_size, :]

            X_test = X[training_size:, :]
            y_test = y[training_size:, :]

            clf = TopKRanker(LogisticRegression())
            clf.fit(X_train, y_train)

            # find out how many labels should be predicted
            top_k_list = list(map(int, y_test.sum(axis=1).T.tolist()[0]))
            preds = clf.predict(X_test, top_k_list)

            results = {}
            averages = ["micro", "macro", "samples", "weighted"]
            for average in averages:
                results[average] = f1_score(y_test, preds, average=average)

            all_results[train_percent].append(results)
    print('Results, using embeddings of dimensionality', X.shape[1])
    print('-------------------')
    print('Train percent:', 'average f1-score')
    x_darwing = []
    y_darwing = []
    for train_percent in sorted(all_results.keys()):
        av = 0
        stder = np.ones(number_shuffles)
        i = 0
        for x in all_results[train_percent]:
            stder[i] = x["micro"]
            i += 1
            av += x["micro"]
        av /= number_shuffles
        x_darwing.append(train_percent)
        y_darwing.append(av)
        print(train_percent, ":", av)

    # drawing
    plt.plot(x_darwing, y_darwing, '^-', color='#F4A460', label="ppr", ms=6, lw=1.8)  
    plt.xlabel("Training Ratio", fontsize=18)  
    plt.ylabel("Micro-F1", fontsize=18)  
    plt.legend(loc="lower right", fontsize=15)  
    plt.tick_params(labelsize=13)
    plt.grid()
    return plt


def parse_args():
    parser = argparse.ArgumentParser(description="Community Discover.")
    parser.add_argument('-label', nargs='?', default='data/PPI.cmty',
                        help='Input label file path')
    parser.add_argument('-emb', nargs='?', default='emb/PPI.emb',
                        help='embeddings file path')
    parser.add_argument('-shuffle', type=int, default=10,
                        help='number of shuffule')
    return parser.parse_args()


if __name__ == '__main__':
    evaluate('/home/star/linlonglong/new_mode/data/BlogCatalog-dataset/data/group-edges.csv',
             '/home/star/linlonglong/new_mode/embeddings/lemane/lemane-BlogCatalog-output.txt', 10, "blog")
