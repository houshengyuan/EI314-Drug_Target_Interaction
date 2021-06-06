#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
from models import *
from utils import *
from main import *
import math

def output_attention_weight():
    smiles=input("please input the SMILES: ")
    target=input("please input the Target: ")
    model_path=input("please input the log dir")
    X_drugs, X_targets, y =[smiles],[target],[1.0]
    train_set, val_set, test_set = data_process(X_drugs, X_targets, y, frac=[0, 0, 1], random_seed=2)
    config = get_config()
    config['visual_attention']=True
    model = MPNN_CNN(**config)
    model = model.to(train_device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, dim=0)
    params = {'batch_size': 1, 'shuffle': False, 'num_workers': config['num_workers'],'drop_last': False}
    testset_generator = data.DataLoader(data_loader(test_set.index.values, test_set.Label.values, test_set, **config),**params)
    model = load_model(model, model_path)
    with torch.no_grad():
        model.eval()
        for i, (v_d, v_p, label) in enumerate(testset_generator):
            v_p = v_p.float().to(train_device)
            score = model(v_d, v_p)
            predictions = torch.max(score.data, 1)[1].detach().cpu().numpy()
            label_ids = label.to('cpu').numpy()
            y_label = label_ids.flatten().tolist()
            y_pred = predictions.flatten().tolist()
            print("The pair has label: ", y_label[0])
            print("The model predicts output: ", y_pred[0])
            print("The model predicts type: ", label_ids[0])
    attention_weight=np.load("attention_weight.npy",allow_pickle=True)
    conv_out=np.load("conv_out.npy",allow_pickle=True)
    attention_weight=attention_weight.flatten()
    max_index=np.argmax(attention_weight)
    print("Calculating.....")
    stride=math.floor((MAX_SEQ_PROTEIN-sum(config['cnn_target_filters'])+3)/100)
    kernel_size=(MAX_SEQ_PROTEIN-sum(config['cnn_target_filters'])+3)-(100-1)*stride
    start1=kernel_size*max_index
    end1=kernel_size*(max_index+1)
    conv_out=np.squeeze(conv_out,0)
    max_index2=np.argmax(conv_out[:,start1,end1],dim=1)
    start2=max_index2+start1-sum(config['cnn_target_filters'])/2
    end2=max_index2+start1+sum(config['cnn_target_filters'])/2
    print("Start index: ",start2)
    print("End index: ",end2)
    print("FASTA subsequence result: ",target[start2:end2])


if __name__ == "__main__":
    output_attention_weight()
