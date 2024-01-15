import time

import yaml
import eva.classifier as cl
from PSNE import PSNE_model


import numpy
from numpy import float16
def _get_parameter_ppr():
    with open('my_file.yaml', 'r') as c:
        config = yaml.safe_load(c)
    dac = config['appr']['alpha']
    window_size = config['appr']['window_size']
    mu = config['appr']['mu']
    rounds = config['appr']['num_round']
    data_path = config['appr']['dataset_path']
    emb_path = config['appr']['save_emb_path']
    graph_name = config['appr']['graph_name']
    return dac, window_size, rounds, emb_path, data_path,graph_name, mu


def _get_parameter_eva():
    with open('my_file.yaml', 'r') as c:
        config = yaml.safe_load(c)
    label_path = config['evaluate']['label_path']
    save_emb_path = config['evaluate']['save_emb_path']
    shuffle = config['evaluate']['shuffle']

    return label_path, save_emb_path, shuffle


def save_embedding(emb_file, features):
    # save node embedding into emb_file with word2vec format
    f_emb = open(emb_file, 'w')
    f_emb.write(str(len(features)) + " " + str(features.shape[1]) + "\n")
    for i in range(len(features)):
        s = str(i) + " " + " ".join(str(f) for f in features[i].tolist())
        f_emb.write(s + "\n")
    f_emb.close()

    
def experiment_go(t):
    dac, window_size, rou, emb_path, data_path,graph_name, mu = _get_parameter_ppr() 
    
    model = PSNE_model(128, window_size=window_size,mu=mu, num_round=rou, worker=10, a_decay=dac)
    emb = model(data_path,dataset_name=graph_name)
   
    save_embedding(emb_path, emb)
    label, emb, shuffle = _get_parameter_eva()
    plt = cl.evaluate(label, emb, shuffle,graph_name)
    
 
    print('done')
    return


if __name__ == '__main__':
    experiment_go(1)
