# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import pickle
from keras.models import Sequential, Model, load_model
from keras.layers.recurrent import GRU
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Dropout
from keras.layers import Input
import csv




class Hyperparams:
    batch_size = 64
    embed_dim = 300
    hidden_dim = 1000
    ctxlen = 100 # For inference
    maxlen = 400 # For training


def build_graph(seqlen):
    sequence = Input(shape=(seqlen,), dtype="int32")
    embedded = Embedding(13, Hyperparams.embed_dim, mask_zero=True)(sequence)
    gru1 = GRU(Hyperparams.hidden_dim, return_sequences=True)(embedded)
    after_dp = Dropout(0.5)(gru1)
    gru2 = GRU(Hyperparams.hidden_dim, return_sequences=True)(after_dp)
    after_dp = Dropout(0.5)(gru2)
    output = TimeDistributed(Dense(13, activation="softmax"))(after_dp)
    
    model = Model(input=sequence, output=output)
    
    return model

def get_the_latest_ckpt():
    import os
    import glob
    
    latest_ckpt = max(glob.glob('ckpt/*.h5'), key=os.path.getctime)
    return latest_ckpt
def create_test_data():
    digit2idx, idx2digit = load_vocab()

    ids = [line.split(',')[0] for line in open('data/file.csv', 'r').read().splitlines()[1:]]
    lines = [line.split('"')[1] + "," for line in open('data/file.csv', 'r').read().splitlines()[1:]]
    xs = []
    for line in lines:
        x = [digit2idx[digit] for digit in line[-Hyperparams.ctxlen:]]
        x = [0] * (Hyperparams.ctxlen - len(x)) + x # zero prepadding
        
        xs.append(x)
         
    X = np.array(xs, np.int32)
    
    pickle.dump((X, ids), open('data/test.pkl', 'wb'))
def some(s):
    with open('data/file.csv', 'w',newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "sequence"])
        z=s
        w.writerow([1,z])
            

	
def load_test_data():
    create_test_data()
    X, ids = pickle.load(open('data/test.pkl', 'rb'))     
    return X, ids

    
def load_vocab():
    vocab = 'E,-0123456789' # E:zero-padding
    digit2idx = {digit:idx for idx, digit in enumerate(vocab)}
    idx2digit = {idx:digit for idx, digit in enumerate(vocab)}
    
    return digit2idx, idx2digit

def callme(s):
    some(s)
    digit2idx, idx2digit = load_vocab()
    X, I = load_test_data()
    model = build_graph(Hyperparams.ctxlen)
    latest_ckpt = get_the_latest_ckpt()
    model.load_weights(latest_ckpt)


    for step in range(0, len(X), Hyperparams.batch_size):
        xs = X[step:step+Hyperparams.batch_size]
        ids = I[step:step+Hyperparams.batch_size]
        _preds = []
        for _ in range(10):
            preds = model.predict(xs)
            preds = preds[:, -1, :] #(None, 13)
            preds = np.argmax(preds, axis=-1) #(None,)
            _preds.append(preds)
            preds = np.expand_dims(preds, -1) #(None, 1)
            xs = np.concatenate((xs, preds), 1)[:, 1:]
        _preds = np.array(_preds).transpose()
        for p, id in zip(_preds, ids):
            p = "".join(idx2digit[idx] for idx in p).split(",")[0]
            return p
            
                
                

