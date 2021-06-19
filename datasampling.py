#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
(1)fetch all positive pairs
(2)sift all unseen data points in negative pairs
(3)To sample negative pairs, we first calculate drugs and protein frequency respectively
in positive pairs, then we assign joint possibility to every negative pair and sort them.
(4)sample negative pairs by softmax of joint possibility
统计一下正样本的数据分布规律
"""

import pandas as pd
import numpy as np

def read_inpd(filepath):
   data=pd.read_csv(filepath,header=0)
   return data


def generate_dict(sample):
   smiles=sample['SMILES'].value_counts(normalize = False, dropna = False)
   fasta=sample['FASTA'].value_counts(normalize = False, dropna = False)
   return smiles, fasta


def sample(data):
   neg = data[data['Label'] == 0]
   pos = data[data['Label'] == 1]
   len_neg = neg.shape[0]
   index=[i*25 for i in range(int(len_neg/25))]
   d_neg = neg.iloc[index]
   d_neg.reset_index(drop=True,inplace=True)
   d = pd.concat([d_neg, pos], axis=0)
   d.columns = ['SMILES', 'FASTA', 'Label']
   d.to_csv('train_100.csv', index=False)


if __name__=="__main__":
   filepath="train.csv"
   data=read_inpd(filepath)
   sample(data)
