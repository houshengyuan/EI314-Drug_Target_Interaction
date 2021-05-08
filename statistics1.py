#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
(1)优先抽取15000条 seen protein seen SMILES 按照频率乘积抽取
(2)而后抽取等量的  有一个unseen的序列
负样本存在着较多的未在正样本中出现的FASTA，但是正样本中的SMILES能够很好地代表整体的样本种类
首先计算正样本的softmax (1+freq_pos)*log(1+freq_pos)
我们可以对所有模型做一个整合，CNN,RNN,LSTM,Transformer,GCN可以换着来
"""

import pickle
import pandas as pd
import numpy as np
log=open("statistics.txt","a+",encoding="UTF-8")
np.random.seed(0)

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


def sampling(data=None):
    #pos=data[data['Label']==1]
    #pos.reset_index(drop=True,inplace=True)
    neg=pd.read_csv("./data/pre_sample.csv")
    #dict1=pos['SMILES'].value_counts(normalize =False, dropna = False).to_dict()
    #dict2=pos['FASTA'].value_counts(normalize =False, dropna = False).to_dict()
    #neg.loc[:,'FD']=0;neg.loc[:,'FT']=0
    #for i in range(len(neg)):
    #  print(i)
    #  neg.loc[i,'FD']=(dict1[neg.loc[i,'SMILES']] if neg.loc[i,'SMILES'] in dict1 else 0)
    #  neg.loc[i,'FT']=(dict2[neg.loc[i,'FASTA']] if neg.loc[i,'FASTA'] in dict2 else 0)
    print("ok1")
    #tmp1=np.array(neg['FD']).reshape(1,-1)
    #tmp2=np.array(neg['FT']).reshape(1,-1)
    print("ok2")
    #weight=(1+tmp1)*(1+tmp2)/np.sum((1+tmp1)*(1+tmp2))
    print("ok3")
    sample=np.random.choice(a=neg.index.values,size=2000,replace=False).tolist()#p=weight.flatten()).tolist()
    print("ok4")
    neg.loc[sample].to_csv("train_neg.csv",mode="w+",index=False)


if __name__=="__main__":
    #file_path="data/pre_sample.csv"
    print("------------------------------------------",file=log)
    #data=read_in(file_path)
    #analysis1(data)
    #analysis2(data)
    #presample(data)
    sampling()
