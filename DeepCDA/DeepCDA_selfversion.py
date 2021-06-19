import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import scipy.io as scio

protein_dict = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
                "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
                "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
                "U": 19, "T": 20, "W": 21, 
                "V": 22, "Y": 23, "X": 24, 
                "Z": 25 }

protein_dict_len = 25

smiles_dict = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
                "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
                "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
                "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
                "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
                "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
                "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
                "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

smiles_dict_len = 64

protein_max_len=1000
smiles_max_len=100


def convert_smile(smile):
    # input    format: list
    # output  format: list  1*100
    smile_length=len(smile)
    if smile_length>100:
        smile_length=100
    result=[]
    for i in range(smile_length):
        if smile[i] not in smiles_dict:
            result.append(0)
        else:
            result.append(smiles_dict[smile[i]])
    for i in range(smile_length,100):
        result.append(0)

    #print("converting smile:{}".format(len(result)))
    return result

def convert_fasta(fasta):
    # input    format: list
    # output  format: list  1*100
    fasta_length=len(fasta)
    if fasta_length>1000:
        fasta_length=1000
    result=[]
    for i in range(fasta_length):
        if fasta[i] not in protein_dict:
            result.append(0)
        else:
            result.append(protein_dict[fasta[i]])
    for i in range(fasta_length,1000):
        result.append(0)

    #print("converting fasta:{}".format(len(result)))
    return result

def affinity2label(affinity):
    affinity=affinity-7.5
    l=1/(1+pow(np.e,affinity))
    if l >=0.5:
        return 1
    else:
        return 0

def label2affinity(label):
    if label==1.0:
        return 1.0
    else:
        return 0.0


def generate_mat():
    # param setting
    TRAINCSVURL='../train/train.csv'
    VALCSVURL='../validation/validation.csv'
    MATURL='datasets/mydataset_folded_v2.mat'
    TRAINSPLITRATIO=0.8                # 80% --> train   20% --> test

    # load .csv dataset
    traindataset=pd.read_csv(TRAINCSVURL)
    traindataset=shuffle(traindataset).values.tolist()
    valdataset=pd.read_csv(VALCSVURL)
    valdataset=shuffle(valdataset).values.tolist()

    # converting process
    length=len(traindataset)
    train_length=int(length*TRAINSPLITRATIO)

    train_folds_drugs=[]
    train_folds_proteins=[]
    train_folds_affinity=[]
    for i in range(train_length):
        train_folds_drugs.append(convert_smile(traindataset[i][0]))
        train_folds_proteins.append(convert_fasta(traindataset[i][1]))
        train_folds_affinity.append(label2affinity(traindataset[i][2]))
       
    test_folds_drugs=[]
    test_folds_proteins=[]
    test_folds_affinity=[]
    for i in range(train_length,length):
        test_folds_drugs.append(convert_smile(traindataset[i][0]))
        test_folds_proteins.append(convert_fasta(traindataset[i][1]))
        test_folds_affinity.append(label2affinity(traindataset[i][2]))

    val_folds_drugs=[]
    val_folds_proteins=[]
    val_folds_affinity=[]
    for i in range(len(valdataset)):
        val_folds_drugs.append(convert_smile(valdataset[i][0]))
        val_folds_proteins.append(convert_fasta(valdataset[i][1]))
        val_folds_affinity.append(label2affinity(valdataset[i][2]))

    scio.savemat(MATURL, {'train_folds_drugs':np.array([train_folds_drugs]),'train_folds_proteins':np.array([train_folds_proteins]),'train_folds_affinity':np.array([train_folds_affinity]),'test_folds_drugs':np.array([test_folds_drugs]),'test_folds_proteins':np.array([test_folds_proteins]),'test_folds_affinity':np.array([test_folds_affinity]),'val_folds_drugs':np.array([val_folds_drugs]),'val_folds_proteins':np.array([val_folds_proteins]),'val_folds_affinity':np.array([val_folds_affinity])})



def generate_test():
    VALCSVURL='../validation/validation.csv'
    MATURL='datasets/mydataset_val.mat'

    valdataset=pd.read_csv(VALCSVURL)
    valdataset=shuffle(valdataset).values.tolist()

    XD_train=[]
    XT_train=[]
    y_train=[]
    for i in range(len(valdataset)):
        XD_train.append(convert_smile(valdataset[i][0]))
        XT_train.append(convert_fasta(valdataset[i][1]))
        y_train.append(label2affinity(valdataset[i][2]))

    scio.savemat(MATURL,{'XD_train':np.array(XD_train),'XT_train':XT_train,'y_train':y_train})


def generate_train():
    TRAINCSVURL='../train/train.csv'
    MATURL='datasets/mydataset_train.mat'

    traindataset=pd.read_csv(TRAINCSVURL)
    traindataset=shuffle(traindataset).values.tolist()

    XD_train=[]
    XT_train=[]
    y_train=[]
    for i in range(len(traindataset)):
        XD_train.append(convert_smile(traindataset[i][0]))
        XT_train.append(convert_fasta(traindataset[i][1]))
        y_train.append(label2affinity(traindataset[i][2]))

    scio.savemat(MATURL,{'XD_train':np.array(XD_train),'XT_train':XT_train,'y_train':y_train})


generate_mat()