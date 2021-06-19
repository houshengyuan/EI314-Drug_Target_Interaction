#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import pickle
np.random.seed(0)
log=open("statistics.txt","a+",encoding="UTF-8")


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


def read_in(filepath):
   data=pd.read_csv(filepath)
   return data


def analysis1(data):
    print("数据的总长度为",len(data),file=log)
    posdata=data[data['Label'].isin([1])]
    negdata=data[data['Label'].isin([0])]
    print("正样本的数量为",len(posdata),file=log)
    print("负样本的数量为",len(negdata),file=log)
    print("样本SMILES数量为",len(set(data['SMILES'])),file=log)
    print("样本FASTA数量为",len(set(data['FASTA'])),file=log)
    print("正样本SMILES数量为",len(set(posdata['SMILES'])),file=log)
    print("负样本SMILES数量为",len(set(negdata['SMILES'])),file=log)
    print("正样本FASTA数量为",len(set(posdata['FASTA'])),file=log)
    print("负样本FASTA数量为",len(set(negdata['FASTA'])),file=log)
    spvalue=posdata["SMILES"].value_counts().values
    snvalue=negdata["SMILES"].value_counts().values
    fpvalue=posdata["FASTA"].value_counts().values
    fnvalue=negdata["FASTA"].value_counts().values
    svalue=data["SMILES"].value_counts().values
    fvalue=data["FASTA"].value_counts().values
    fw=open("res1.txt","wb+")
    result=[]
    for t in [spvalue,snvalue,fpvalue,fnvalue,svalue,fvalue]:
      result.append(dict(zip(*np.unique(t, return_counts=True))))
    pickle.dump(result,fw)


def analysis2(data):
    pos=data[data['Label']==1]
    pos.reset_index(drop=True,inplace=True)
    neg=data[data['Label']==0]
    neg.reset_index(drop=True,inplace=True)
    dict1=pos['SMILES'].value_counts(normalize = False, dropna = False).to_dict()
    dict2=pos['FASTA'].value_counts(normalize = False, dropna = False).to_dict()
    two_unseen=0;two_seen=0;protein_unseen=0;SMILES_unseen=0
    for i in range(len(neg)):
      flag1=(neg.loc[i,'SMILES'] in dict1)
      flag2=(neg.loc[i,'FASTA'] in dict2)
      if  flag1 and flag2:
        two_seen+=1
      elif not flag1 and not flag2:
        two_unseen+=1
      elif not flag1 and flag2:
        SMILES_unseen+=1
      else:
        protein_unseen+=1
    print("Seen:",two_seen,file=log)
    print("Protein_UnSeen:",protein_unseen,file=log)
    print("SMILES_UnSeen:",SMILES_unseen,file=log)
    print("UnSeen:",two_unseen,file=log)


def presample(data):
    pos=data[data['Label']==1]
    pos.reset_index(drop=True,inplace=True)
    neg=data[data['Label']==0]
    neg.reset_index(drop=True,inplace=True)
    pre_sample=np.random.choice(a=neg.index.values,size=1000000,replace=False,p=np.ones(len(neg))/len(neg)).tolist()
    tmp=neg.loc[pre_sample]
    print(len(tmp))
    tmp.reset_index(drop=True,inplace=True)
    tmp.to_csv("pre_sample.csv",mode="w+",index=False)


def data_split(data=None):
    data=data.sample(frac=1).reset_index(drop=True)
    size=len(data)
    for i in range(10):
      print(i)
      if i<9:
        df=data.loc[i*(size/10):(i+1)*(size/10)]
      else:
        df=data.loc[i*(size/10):-1]
      df.reset_index(drop=True).to_csv("robust/"+"test_set"+str(i)+".csv",index=0)


if __name__=="__main__":
   filepath="train.csv"
   data=read_inpd(filepath)
   sample(data)
