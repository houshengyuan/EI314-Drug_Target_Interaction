#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from utils import *
import matplotlib.pyplot as plt
import time
from torch.utils.data import SequentialSampler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from models import device as train_device, MPNN_CNN
from torch import nn

torch.manual_seed(1)
np.random.seed(1)
log_dir = os.path.join('log', time.asctime(time.localtime(time.time()))).replace(" ", "_").replace(":", "_")


def get_config():
    config = {}
    config['input_dim_drug'] = 1024
    config['input_dim_protein'] = 8420
    config['hidden_dim_drug'] = 128  # hidden dim of drug
    config['hidden_dim_protein'] = 256  # hidden dim of protein
    config['cls_hidden_dims'] = [200, 100]  # decoder classifier dim 1

    config['batch_size'] = 256
    config['train_epoch'] = 100
    config['LR'] = 0.005
    config['num_workers'] = 4
    config['attention']=True
    config['mpnn_hidden_size'] = 128
    config['mpnn_depth'] = 3

    config['cnn_target_filters'] = [16, 32, 48]
    config['cnn_target_kernels'] = [24, 48, 72]

    config['modelpath'] = "model"
    config['visual_attention']=False
    config['concatenation']=True
    return config


def plot(train_epoch, acc_record, f1_record, precision_record, recall_record, loss_record, train_acc_record,
         train_f1_record, train_precision_record, train_recall_record, train_loss_record):
    # plot five statistic .png picture
    x = np.arange(1, train_epoch + 1)
    plt.plot(x, np.array(train_acc_record), label="train")
    plt.plot(x, np.array(acc_record), label="validation")
    plt.title("model accuracy")
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'acc.png'))
    plt.clf()

    plt.plot(x, np.array(train_f1_record), label="train")
    plt.plot(x, np.array(f1_record), label="validation")
    plt.title("model F1")
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'f1.png'))
    plt.clf()

    plt.plot(x, np.array(train_precision_record), label="train")
    plt.plot(x, np.array(precision_record), label="validation")
    plt.title("model precision")
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'precision.png'))
    plt.clf()

    plt.plot(x, np.array(train_recall_record), label="train")
    plt.plot(x, np.array(recall_record), label="validation")
    plt.title("model recall")
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'recall.png'))
    plt.clf()

    plt.plot(x, np.array(train_loss_record), label="train")
    plt.plot(x, np.array(loss_record), label="validation")
    plt.title("model loss")
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'loss.png'))
    plt.clf()


def save_model(model, path_dir, **config):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    torch.save(model.module.state_dict(), path_dir + '/model.pt')
    save_dict(path_dir, config)


def load_model(model, path_dir):
    para = torch.load(path_dir + '/model.pt')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, dim=0)
        model.load_state_dict(para)
    else:
        model.load_state_dict({k.replace('module.', ''): v for k, v in para.items()})
    return model


def test(device, data_generator, model):
    predicted_y = []
    true_y = []
    with torch.no_grad():
        model.eval()
        for i, (d, p, label) in enumerate(data_generator):
            p = p.float().to(device)
            pred = model(d, p)
            predictions = torch.max(pred.data, 1)[1].detach().cpu().numpy()
            label_ids = label.to('cpu').numpy()
            true_y += label_ids.flatten().tolist()
            predicted_y += predictions.flatten().tolist()
        pred_res = pd.DataFrame(predicted_y)
    model.train()
    return pred_res, accuracy_score(true_y, predicted_y), precision_score(true_y, predicted_y), recall_score(true_y,
                                                                                                             predicted_y), f1_score(
        true_y, predicted_y)


def train(model, device, train_set, val_set, test_set, **config):
    lr = config['LR']
    batch_size = config['batch_size']
    train_epoch = config['train_epoch']
    model = model.to(device)
    if 'num_workers' not in config.keys():
        config['num_workers'] = 0
    # split the workload to all cuda evenly
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, dim=0)
    # dynamically change the lr
    # for every 5 epoches  lr:=lr*0.8
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    miles = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
             65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=miles, gamma=0.8)
    # data loader for train val and test(if val and test existes)
    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': config['num_workers'], 'drop_last': False}
    trainset_generator = data.DataLoader(data_loader(train_set.index.values, train_set.Label.values, train_set, **config), **params)
    validset_generator = data.DataLoader(data_loader(val_set.index.values, val_set.Label.values, val_set, **config), **params)
    info = data_loader(test_set.index.values, test_set.Label.values, test_set, **config)
    params_test = {'batch_size': batch_size,
                   'shuffle': False,
                   'num_workers': config['num_workers'],
                   'drop_last': False,
                   'sampler': SequentialSampler(info)}
    testing_generator = data.DataLoader(data_loader(test_set.index.values, test_set.Label.values, test_set, **config),**params_test)
    # recode the metrics when training
    train_acc_record, train_f1_record, train_precision_record, train_recall_record, train_loss_record = [], [], [], [], []
    acc_record, f1_record, precision_record, recall_record, loss_record = [], [], [], [], []
    start = time.time()
    for epo in range(train_epoch):
        loss_val = 0
        for i, (d, p, label) in enumerate(trainset_generator):
            #print(i)
            p = p.float().to(device)
            # print(d.size,p.size)
            pred = model(d, p)
            label = Variable(torch.from_numpy(np.array(label)).long()).to(device)
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(pred, label)
            loss_val += loss.item() * label.size(0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()
        train_loss_record.append(loss_val)
        tmp = time.time()
        # Output the training process
        print(' Epoch: ' + str(epo + 1) + '  Loss ' + str(loss_val) + ". Consumed Time " + str(
            int(tmp - start) / 60) + " mins", file=open(os.path.join(log_dir, 'log.txt'), 'a+'), flush=True)
        print(' Epoch: ' + str(epo + 1) + '  Loss ' + str(loss_val) + ". Consumed Time " + str(int(tmp - start) / 60) + " mins", flush=True)
        start = tmp
        # test current model
        with torch.set_grad_enabled(False):
            _, accuracy, precision, recall, f1 = test(device, trainset_generator, model)
            print('Training at Epoch ' + str(epo + 1) + ', Accuracy: ' + str(accuracy) + ', Precision: ' + str(
                precision) + ', Recall: ' + str(recall) + ' , F1: ' + str(f1),
                  file=open(os.path.join(log_dir, 'log.txt'), 'a+'), flush=True)
            print('Training at Epoch ' + str(epo + 1) + ', Accuracy: ' + str(accuracy) + ', Precision: ' + str(
                precision) + ', Recall: ' + str(recall) + ' , F1: ' + str(f1), flush=True)
            train_acc_record.append(accuracy)
            train_f1_record.append(f1)
            train_precision_record.append(precision)
            train_recall_record.append(recall)
            _, accuracy, precision, recall, f1 = test(device, validset_generator, model)
            print('Validation at Epoch ' + str(epo + 1) + ', Accuracy: ' + str(accuracy) + ', Precision: ' + str(
                precision) + ', Recall: ' + str(recall) + ' , F1: ' + str(f1), file=open(os.path.join(log_dir, 'log.txt'), 'a+'), flush=True)
            print('Validation at Epoch ' + str(epo + 1) + ', Accuracy: ' + str(accuracy) + ', Precision: ' + str(
                precision) + ', Recall: ' + str(recall) + ' , F1: ' + str(f1), flush=True)
            acc_record.append(accuracy)
            f1_record.append(f1)
            precision_record.append(precision)
            recall_record.append(recall)
            lloss = 0
            for i, (d, p, label) in enumerate(validset_generator):
                p = p.float().to(device)
                pred = model(d, p)
                label = Variable(torch.from_numpy(np.array(label)).long()).to(device)
                loss_fct = torch.nn.CrossEntropyLoss()
                loss_ = loss_fct(pred, label)
                lloss += loss_.item() * label.size(0)
            loss_record.append(lloss)
    plot(train_epoch, acc_record, f1_record, precision_record, recall_record, loss_record, train_acc_record,
         train_f1_record, train_precision_record, train_recall_record, train_loss_record)
    pred_res, accuracy, precision, recall, f1 = test(device, testing_generator, model)
    print('Test at Epoch ' + str(epo + 1) + ' , Accuracy: ' + str(accuracy) + ', Precision:' + str(
        precision) + ' , Recall: ' + str(recall) + ' , F1: ' + str(f1),
          file=open(os.path.join(log_dir, 'log.txt'), 'a+'), flush=True)
    print('Test at Epoch ' + str(epo + 1) + ' , Accuracy: ' + str(accuracy) + ', Precision:' + str(
        precision) + ' , Recall: ' + str(recall) + ' , F1: ' + str(f1), flush=True)
    pred_res.to_csv(os.path.join(log_dir, 'predicted_labels.csv'))
    save_model(model, log_dir, **config)


def robustness_test():
    X_drugs, X_targets, y = read_file_training_dataset_drug_target_pairs('../train/train_new.csv')
    train_set, val_set, test_set = data_process(X_drugs, X_targets, y, frac=[0, 0, 1], random_seed=2)
    config = get_config()
    model = MPNN_CNN(**config)
    model = model.to(train_device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, dim=0)
    params = {'batch_size': config['batch_size'], 'shuffle':True, 'num_workers': config['num_workers'], 'drop_last': False}
    testset_generator = data.DataLoader(data_loader(test_set.index.values, test_set.Label.values, test_set, **config), **params)
    model=load_model(model,"model")
    y_pred = []
    y_label = []
    with torch.no_grad():
        model.eval()
        for i, (v_d, v_p, label) in enumerate(testset_generator):
            if i % 10 == 0:
                print("epoch: ", i)
            v_p = v_p.float().to(device)
            score = model(v_d, v_p)
            predictions = torch.max(score.data, 1)[1].detach().cpu().numpy()
            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + predictions.flatten().tolist()
    return accuracy_score(y_label, y_pred), precision_score(y_label, y_pred), recall_score(y_label, y_pred), f1_score(y_label, y_pred)


def main():
    X_drugs, X_targets, y = read_file_training_dataset_drug_target_pairs('../train/train_new.csv')
    train_set, val_set, test_set = data_process(X_drugs, X_targets, y, frac=[0.8, 0.1, 0.1], random_seed=2)
    '''
    config = generate_config(drug_encoding='MPNN', target_encoding='CNN',
                             cls_hidden_dims=[200, 100], train_epoch=100, LR=1e-4,
                             batch_size=256, hidden_dim_drug=128,
                             hidden_dim_protein=256, preTrain=True,
                             mpnn_hidden_size=128, mpnn_depth=3,
                             cnn_target_filters=[16, 32, 48], cnn_target_kernels=[24, 48, 72], num_workers=4)
    '''
    config = get_config()
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    model = MPNN_CNN(**config)
    train(model, train_device, train_set, val_set, test_set, **config)


if __name__ == '__main__':
    main()
    #robustness_test()
