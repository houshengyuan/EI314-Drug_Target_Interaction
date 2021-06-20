#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pandas as pd
from MAC.main import *
import argparse
from torch import nn

def argparser():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model_dir',
      type=str,
      default='MAC/model',
      help='Path to the model dir (a directory)'
  )
  parser.add_argument(
     '--test_path',
     type=str,
     default='test.csv',
     help='Path to the test dataset (a .csv file)'
  )
  parser.add_argument(
      '--result_save_path',
      type=str,
      default='result.csv',
      help='Path to the result file (a .csv file)'
  )
  flags, unparsed = parser.parse_known_args()
  return flags


def robustness_test(flags):
   #load in and
   X_drugs, X_targets, y = read_file_training_dataset_drug_target_pairs(flags.test_path)
   train_set, val_set, test_set = data_process(X_drugs, X_targets, y, frac=[0, 0, 1], random_seed=2, aug=False)

   #configuration settings
   config = get_config()

   #define model
   config['concatenation'] = False
   config['attention'] = True
   model = MPNN_CNN(**config)
   model = model.to(train_device)
   if torch.cuda.device_count() > 1:
      model = nn.DataParallel(model, dim=0)

   params = {'batch_size': config['batch_size'], 'shuffle': True, 'num_workers': config['num_workers'],'drop_last': False}

   #generate testing data
   testset_generator = data.DataLoader(data_loader(test_set.index.values, test_set.Label.values, test_set, **config),**params)

   #load pretrained model and test on candidate dataset
   model = load_model(model, flags.model_dir)
   y_pred = []
   y_label = []
   with torch.no_grad():
      model.eval()
      for i, (v_d, v_p, label) in enumerate(testset_generator):
         v_p = v_p.float().to(train_device)
         score = model(v_d, v_p)
         predictions = torch.max(score.data, 1)[1].detach().cpu().numpy()
         label_ids = label.to('cpu').numpy()
         y_label = y_label + label_ids.flatten().tolist()
         y_pred = y_pred + predictions.flatten().tolist()

   #output evaluation metrics
   print("accuracy score: ", accuracy_score(y_label, y_pred))
   print("precision: ", precision_score(y_label, y_pred))
   print("recall: ", recall_score(y_label, y_pred))
   print("F1 score:", f1_score(y_label, y_pred))

   #save the prediction result
   print("Saving...")
   result=pd.DataFrame({'Accuracy':float(accuracy_score(y_label, y_pred)),'F1':float(f1_score(y_label, y_pred))},index=[0])
   result.to_csv(flags.result_save_path,index=False)
   print("Successfully saved in "+flags.result_save_path)


if __name__ == "__main__":
    flags=argparser()
    robustness_test(flags)
