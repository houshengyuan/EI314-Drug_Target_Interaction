#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import train_model
from utils import *



def main():
    X_drugs, X_targets, y = read_file_training_dataset_drug_target_pairs('../train/train.csv')
    train, val, test = data_process(X_drugs, X_targets, y,drug_encoding='MPNN', target_encoding='CNN',frac=[0.8,0.1,0.1],random_seed = 2)
    config = generate_config(drug_encoding = 'MPNN',target_encoding = 'CNN',
                            cls_hidden_dims = [200,100],train_epoch = 5,
                            test_every_X_epoch=10,LR =1e-4,
                            batch_size = 256,hidden_dim_drug = 128,
                            hidden_dim_protein=256,
                            mpnn_hidden_size = 128,mpnn_depth = 3,
                            cnn_target_filters = [16,32,48],cnn_target_kernels = [24,48,72],num_workers=4)
    model = train_model.model_initialize(**config)
    model.train(train, val, test)
    model.save_model("model")


if __name__=='__main__':
    main()
