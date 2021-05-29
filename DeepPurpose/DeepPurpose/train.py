#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import DTI as models
import utils,dataset
import warnings
warnings.filterwarnings("ignore")

X_drugs, X_targets, y = dataset.read_file_training_dataset_drug_target_pairs('../train/train.csv')
train, val, test = utils.data_process(X_drugs, X_targets, y,
                                drug_encoding='MPNN', target_encoding='CNN',
                                frac=[0.8,0.1,0.1],random_seed = 1)
config = utils.generate_config(drug_encoding = 'MPNN',target_encoding = 'CNN',
                         cls_hidden_dims = [1024,1024,512],train_epoch = 100,
                         test_every_X_epoch=10,LR = 5e-4,
                         batch_size = 256,hidden_dim_drug = 128,
                         mpnn_hidden_size = 128,mpnn_depth = 3,
                         cnn_target_filters = [32,64,96],cnn_target_kernels = [4,8,12])
model = models.model_initialize(**config)
model.train(train, val, test)