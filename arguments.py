#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os

params={
'seq_window_lengths':12,
'smi_window_lengths':8,
'num_windows':32,
'max_seq_len':1000,
'max_smi_len':100,
'learning_rate':2.5e-4,
'num_epoch':50,
'batch_size':128,
'train_path':"..\\train",
'validation_path':"..\\validation",
'log_dir':".\\log",
'drop_rate':0.2
}


def logging(msg):
  fpath = os.path.join(params['log_dir'], "log.txt")
  print("%s\n" % str(msg),file=open( fpath, "a+" ),flush=True,end=" ")

