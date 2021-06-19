#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from main import *

def robustness_test():
   X_drugs, X_targets, y = read_file_training_dataset_drug_target_pairs('robust/test_set0.csv')
   train_set, val_set, test_set = data_process(X_drugs, X_targets, y, frac=[0, 0, 1], random_seed=2, aug=False)
   config = get_config()
   config['concatenation'] = False
   config['attention'] = True
   model = MPNN_CNN(**config)
   model = model.to(train_device)
   if torch.cuda.device_count() > 1:
      model = nn.DataParallel(model, dim=0)
   params = {'batch_size': config['batch_size'], 'shuffle': True, 'num_workers': config['num_workers'],
             'drop_last': False}
   testset_generator = data.DataLoader(data_loader(test_set.index.values, test_set.Label.values, test_set, **config),
                                       **params)
   model = load_model(model, "model")
   y_pred = []
   y_label = []
   with torch.no_grad():
      model.eval()
      for i, (v_d, v_p, label) in enumerate(testset_generator):
         if i % 10 == 0:
            print("epoch: ", i)
         v_p = v_p.float().to(train_device)
         score = model(v_d, v_p)
         predictions = torch.max(score.data, 1)[1].detach().cpu().numpy()
         label_ids = label.to('cpu').numpy()
         y_label = y_label + label_ids.flatten().tolist()
         y_pred = y_pred + predictions.flatten().tolist()
   print("accuracy score: ", accuracy_score(y_label, y_pred))
   print("precision: ", precision_score(y_label, y_pred))
   print("recall: ", recall_score(y_label, y_pred))
   print("F1 score:", f1_score(y_label, y_pred))


if __name__ == "__main__":
    robustness_test()
