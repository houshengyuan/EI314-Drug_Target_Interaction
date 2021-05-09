#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from SMILES import *
from FASTA import *
import os
import numpy as np
import random as rn
from keras import backend as K
from keras.callbacks import History
from arguments import *
import time

os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(0)
rn.seed(0)
print('GPU:', tf.test.is_gpu_available())
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)


tf.set_random_seed(0)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
history = History()

log_dir = os.path.join(params['log_dir'],time.asctime(time.localtime(time.time())).replace(" ", "_").replace(":", "_"))
filepath=os.path.join(log_dir,"log.txt")

if __name__ == "main":
    pass
