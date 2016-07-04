#aftersome long thinking, finally we come to a method which seems to be good and implementable
import os
import random
import theano

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import RMSprop
from keras.models import model_from_json

import numpy as np
import sys
import random
import pandas as pd

from config import tick_use_num, fea_num
from dqn_eval import load_data

def prepare_model():
    model = Sequential()

    model.add(Convolution2D(32, 16, 3, border_mode = 'valid', input_shape=(1, tick_use_num, fea_num), activation = 'relu', subsample = (8, 1)))
    model.add(Convolution2D(32, 6, 2, activation = 'relu', subsample = (2, 1)))
    model.add(Convolution2D(32, 6, 3, activation = 'relu', subsample = (2, 1)))

    model.add(Flatten())
    model.add(Dense(64, activation = 'relu'))

    model.add(Dense(3, activation ='linear'))
    my = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    model.compile(loss='mse', optimizer = my)

    print 'model ok'

    return model

def load_pretrained_model():
    model = model_from_json(open('_model').read())
    model.load_weights('_weights.h5')
    my = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08)
    model.compile(loss='mse', optimizer = my)

    print len(model.layers)
    return model

def prepare_sample_data():
    from config import get_para
    para = get_para('al1607')
    from cnn_eval import of_0516, prepare_data
    contract = 'al1607'
    date = of_0516
    pa = {'train_dates' : date['train'], 'test_dates' : date['test'], 'factors' : [para['kdj_n'], para['macd_n'], para['macd_m'], para['macd_g'], para['bollinger_n'], para['vpin_v'] , False], 'prefix' : contract + '_' + date['name']}
    prepare_data(contract, pa['train_dates'], pa['test_dates'], *pa['factors'])

def the_comm(pos, new_pos, oc, cc):
    if pos == new_pos:
        return 0.0
    if pos == 0:
        return abs(new_pos) * oc

    if pos < 0:
        if new_pos < pos:
            return (pos - new_pos) * oc
        if new_pos > 0:
            return abs(pos) * cc + new_pos * oc
        return (pos - new_pos) * cc
    
    if pos > 0:
        if new_pos > pos:
            return (new_pos - pos) * oc
        if new_pos < 0:
            return abs(new_pos) * oc + pos * cc
        return (pos - new_pos) * cc
            
def new_pos_from_pred(pos, pred):

    arr = np.exp(pred)
    arr = arr / np.sum(arr)
    choice = np.random.choice(range(len(arr)), p = arr)
    new_pos = -2
    if choice == 0:
        new_pos = min(pos + 1, 1)
    if choice == 1:
        new_pos = pos
    if choice == 2:
        new_pos = max(pos - 1, -1)

    return new_pos, choice
    
    
def trade_sequence(datasets, indx, model, tick_size, oc, cc):
    status = {'pos' : 0, 'pnl' : 0.0,
              'b_count' : 0.0, 'b_price' : 0.0,
              's_count' : 0.0, 's_price' : 0.0,
              'comm' : 0.0}
    pnl_seq = []
    features = []
    labels = []
    choice_seq = []
    #print indx, len(datasets)
    for i in range(indx['start'], indx['end']):
        cur = datasets[indx['id']][i]
        pred = model.predict( np.array([[cur['data']]], np.float32) )[0]

        pos = status['pos']
        new_pos, choice = new_pos_from_pred(pos, pred)
        
        features.append([cur['data']])
        one_label = [0.0, 0.0, 0.0]
        one_label[choice] = 1.0
        if i % 1000 == 0:
            print pred, one_label
            
        labels.append(one_label)
        
        choice_seq.append(choice)

        
        status['pos'] = new_pos
        b_count = new_pos - pos if new_pos > pos else 0
        s_count = pos - new_pos if new_pos < pos else 0
        status['b_count'] += b_count
        status['s_count'] += s_count
        status['b_price'] += b_count * float(cur['BestAskPrice'])
        status['s_price'] += s_count * float(cur['BestBidPrice'])
        status['comm'] += the_comm(pos, new_pos, oc, cc)
        status['pnl'] = float(cur['LastPrice']) * status['b_count'] - status['b_price'] + status['s_price'] - float(cur['LastPrice']) * status['s_count'] - status['comm']
        pnl_seq.append(status['pnl'])
    print status
    #print choice_seq
    print labels[-1]
    return status, pnl_seq, features, np.multiply(labels, np.sign(status['pnl']))

def test_indexs(datasets):
    ind = []
    for k in datasets:
        ind.append({'id' : k, 'start' : 0, 'end' : len(datasets[k])})
    return ind

def train_indexes(datasets, length = 1000, step = 180):
    ind = []
    for k in datasets:
        ll = len(datasets[k])
        print k, ll
        start = 0
        end = min(start + length, ll)
        ind.append({'id' : k, 'start' : start, 'end' : end})
        while end < ll:
            start += step
            end = min(start + length, ll)
            ind.append({'id' : k, 'start' : start, 'end' : end})
        #print ind
        #raw_input()
    rand_ind = np.random.permutation(len(ind))
    
    return [ind[i] for i in rand_ind]

def gp_train(model, train_data, train_indx, test_data, test_indx, tick_size = 5.0, oc = 3.08 / 5.0, cc = 0.04 / 5.0):
    feature_buffer = []
    label_buffer = []
    train_times = 0
    train_samples = 0
    for rd in range(1, 80):
        for tr_indx in train_indx:
            status, pnl_seq, feas, ls = trade_sequence(train_data, tr_indx, model, tick_size, oc, cc)
            if status['pnl'] > 0.0 or (status['pnl'] < 0.0 and np.random.rand() < 0.2):
                feature_buffer.extend(feas)
                label_buffer.extend(ls)
                train_samples += len(feas)
            if len(feature_buffer) > 20000:
                train_times += 1
                rand_ind = np.random.permutation(len(feature_buffer))

                feature_buffer = [feature_buffer[i] for i in rand_ind]
                label_buffer = [label_buffer[i] for i in rand_ind]

                train_d = np.array(feature_buffer, np.float32)
                train_l = np.array(label_buffer, np.float32)
                model.fit(train_d, train_l, batch_size = 40, nb_epoch = 1)
                feature_buffer = []
                label_buffer = []
                
                if train_times % 5 == 0:
                    for te_indx in test_indx:
                        te_st, te_pn, te_fea, te_l = trade_sequence(test_data, te_indx, model, tick_size, oc, cc)
                        print te_indx
                        print te_st
                        print pd.Series(te_pn).describe()
                        print pd.Series(te_pn).diff().describe()
                    
                    #raw_input()
                
            
if __name__ == '__main__':
    model = load_pretrained_model()
    #prepare_sample_data()

    train_data = load_data('cnn_train')
    train_indx = train_indexes(train_data)
    print len(train_indx)
    test_data = load_data('cnn_test')
    test_indx = test_indexs(test_data)
    print len(test_indx)
    
    gp_train(model, train_data, train_indx, test_data, test_indx)
