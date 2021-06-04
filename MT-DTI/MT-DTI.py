#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from __future__ import print_function

import keras_metrics
import numpy as np
import tensorflow as tf
import random as rn
from arguments import *
from keras import backend as K
import os
import keras
from keras.callbacks import History
from datahelper import *
from itertools import product
from keras.models import Model
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
import keras_metrics as km
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import BatchNormalization
from keras.layers import Conv2D, GRU
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Masking, RepeatVector, Flatten
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers, layers
from sklearn import metrics
import sys, pickle, os
import math, json, time
import decimal
from random import shuffle
from copy import deepcopy
from sklearn import preprocessing
from tensorflow import metrics
import pandas as pd
from visualization import *

os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(0)
rn.seed(0)
print('GPU:', tf.test.is_gpu_available())
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

tf.set_random_seed(0)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
history = History()

log_dir = os.path.join(params['log_dir'],
                       time.asctime(time.localtime(time.time())).replace(" ", "_").replace(":", "_"))
filepath=os.path.join(log_dir,"log.txt")


def cindex_score(y_true, y_pred):
    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)
    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)
    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)
    return tf.where(tf.equal(g, 0), 0.0, g/f) #select

def build_combined_categorical(params, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
    XDinput = Input(shape=(params['max_smi_len'],), dtype='int32')
    XTinput = Input(shape=(params['max_seq_len'],), dtype='int32')
    encode_smiles = Embedding(input_dim=len(CHARISOSMISET) + 1, output_dim=128, input_length=params['max_smi_len'])(
        XDinput)
    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)
    encode_protein = Embedding(input_dim=len(CHARPROTSET) + 1, output_dim=128, input_length=params['max_seq_len'])(
        XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)
    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein])
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.25)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.25)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)
    predictions = Dense(1, activation='sigmoid')(FC2)
    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])
    interactionModel.compile(optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
                             loss='binary_crossentropy',
                             metrics=[keras.metrics.binary_accuracy,
                                      keras_metrics.binary_precision(),
                                      keras_metrics.binary_recall(),
                                      keras_metrics.f1_score(),
                                      keras_metrics.true_positive(),
                                      keras_metrics.true_negative(),
                                      keras_metrics.false_positive(),
                                      keras_metrics.false_negative()
                                      ])
    print(interactionModel.summary(),file=open(filepath,'a+'),flush=True)
    plot_model(interactionModel, to_file=os.path.join(log_dir,'build_combined_categorical.png'))
    return interactionModel


def general_nfold_cv(runmethod, params):
    param1value = params['num_windows']
    param2value = params['smi_window_lengths']
    param3value = params['seq_window_lengths']
    epoch = params['num_epoch']
    batchsz = params['batch_size']
    train_drugs, train_prots, train_Y = prepare_interaction_pairs(os.path.join(params['train_path'], "train.csv"),
                                                                  type="train", params=params)
    val_drugs, val_prots, val_Y = prepare_interaction_pairs(os.path.join(params['validation_path'], "validation.csv"),
                                                            type="val", params=params)
    gridmodel = runmethod(params, param1value, param2value, param3value)
    #callbacks = [EarlyStopping(monitor='val_loss',patience=15,verbose=2)]
    print(train_drugs.shape, train_prots.shape, train_Y.shape)
    gridres = gridmodel.fit(x=[train_drugs, train_prots], y=train_Y, batch_size=batchsz, epochs=epoch, shuffle=True,
                             validation_data=([val_drugs, val_prots], val_Y), verbose=2)
    predicted_labels = gridmodel.predict([val_drugs, val_prots])
    json.dump(predicted_labels.tolist(), open(os.path.join(log_dir,"predicted_labels.txt"), "w+"))
    tperf= gridmodel.evaluate(([train_drugs, train_prots]), train_Y, verbose=0)
    rperf = gridmodel.evaluate(([val_drugs, val_prots]), val_Y, verbose=2)
    pickle.dump(gridres.history,open(os.path.join(log_dir,"result.pkl"),"wb+"))
    plotLoss(gridres.history, log_dir)
    gridmodel.save(os.path.join(log_dir,"model.h5"))
    return rperf,tperf


def prepare_interaction_pairs(datapath, params, type="train"):
    """
    :param datapath: csv file path of training/validation data
    :return: 返回[drug] [target] [affinity]
    """
    data = pd.read_csv(datapath, encoding='utf-8')
    drugs = handle_sequence(data['SMILES'].tolist(), type="SMILES", params=params)
    targets = handle_sequence(data['FASTA'].tolist(), type="FASTA", params=params)
    affinity = np.array(data['Label'])
    drugcount = len(set(data['SMILES'].tolist()))
    targetcount = len(set(data['FASTA'].tolist()))
    if type == "train":
        print("train drugs: ", drugcount)
        print("train targets: ", targetcount)
    elif type == "val":
        print("val drugs: ", drugcount)
        print("val targets: ", targetcount)
    return drugs, targets, affinity


def datasplit(params):
    pos = pd.read_csv("data\\train_pos.csv", encoding='utf-8')
    neg = pd.read_csv("data\\train_neg.csv", encoding='utf-8')
    neg.drop(columns=['FD', 'FT'], inplace=True)
    train_data_pos = pos.sample(frac=0.9, replace=False, random_state=0, axis=0)
    val_data_pos = pos[~pos.index.isin(train_data_pos.index)]
    train_data_neg = neg.sample(frac=0.9, replace=False, random_state=0, axis=0)
    val_data_neg = neg[~neg.index.isin(train_data_neg.index)]
    train_data = pd.concat([train_data_pos, train_data_neg])
    val_data = pd.concat([val_data_pos, val_data_neg])
    train_data.to_csv(os.path.join(params['train_path'], "train.csv"), index=False, encoding='utf-8')
    val_data.to_csv(os.path.join(params['validation_path'], "validation.csv"), index=False, encoding='utf-8')


def experiment(params, deepmethod):
    """
    perfmeasure: function that takes as input a list of correct and predicted outputs, and returns performance
    higher values should be better, so if using error measures use instead e.g. the inverse -error(Y, P)
    :param params:
    :param deepmethod:
    :return:
    """
    datasplit(params)
    valperf,trainperf = general_nfold_cv(deepmethod, params)
    print("---FINAL RESULTS-----",file=open(filepath,'a+'),flush=True)
    print("Validation Performance ",file=open(filepath,'a+'),flush=True)
    metrics = ['Loss', 'accuracy', 'precision', 'recall', 'F1', 'TP', 'TN', 'FP', 'FN']
    for i in range(len(valperf)):
        print(metrics[i] + ": " + str(valperf[i]),file=open(filepath,'a+'),flush=True)
    print("Train Performance ", file=open(filepath, 'a+'), flush=True)
    for i in range(len(trainperf)):
        print(metrics[i] + ": " + str(trainperf[i]), file=open(filepath, 'a+'), flush=True)
    return valperf,trainperf


if __name__ == "__main__":
   for d in [0.2]:#[0.05,0.1,0.15,0.2,0.25,0.3,0.35]:
    params['drop_rate']=d
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print(params,file=open(filepath,'a+'),flush=True)
    experiment(params, build_combined_categorical)