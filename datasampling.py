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

def sift(data,SMILES,FASTA):
   seen=pd.DataFrame(columns=["SMILES","FASTA"])
   proseen=pd.DataFrame(columns=["SMILES","FASTA"])
   smilesseen=pd.DataFrame(columns=["SMILES","FASTA"])
   unseen=pd.DataFrame(columns=["SMILES","FASTA"])
   for i,data in enumerate(data):
      if data['Label'][i]==0:
        if data['SMILES'][i] in SMILES.keys() and data['FASTA'][i] in FASTA.keys():
           seen.append(data.append(SMILES[data[0]]*FASTA[data[1]]))
        elif data['SMILES'][i] in SMILES.keys() and data['FASTA'][i] not in FASTA.keys():
           remain.append(data.append(SMILES[data[0]]*FASTA[data[1]]))
        elif data['SMILES'][i] not in SMILES.keys() and data['FASTA'][i] in FASTA.keys():
           remain.append()
        else:
           remain.append()
   remain.to_csv("all_seen.csv")
   return remain

def sample(data):
   distribution=np.array(data["jointposs"].tolist())/np.sum(np.array(data["jointposs"].tolist()))


if __name__=="__main__":
   filepath="../train_pos.csv"
   positive_sample=read_inpd(filepath)
   SMILES,FASTA=generate_dict(positive_sample)
   filepath2="../train.csv"
   negative_sample=read_inpd(filepath2)
   remain=sift(negative_sample,SMILES,FASTA)
   #sample(remain)
