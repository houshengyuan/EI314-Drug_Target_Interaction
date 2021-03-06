#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from MAC.utils import *
import matplotlib.pyplot as plt
import time
from torch.utils.data import SequentialSampler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from MAC.models import device as train_device, MPNN_CNN
import argparse

#random seed and log file
torch.manual_seed(1)
np.random.seed(1)
log_dir = os.path.join('log', time.asctime(time.localtime(time.time()))).replace(" ", "_").replace(":", "_")


def argparser():
    """
    parameters of model from command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dim_drug',
        type=int,
        default=1024,
        help='input dim drug'
    )
    parser.add_argument(
        '--input_dim_protein',
        type=int,
        default=8420,
        help='input dim protein'
    )
    parser.add_argument(
        '--hidden_dim_drug',
        type=int,
        default=128,
        help='hidden dim drug'
    )
    parser.add_argument(
        '--hidden_dim_protein',
        type=int,
        default=256,
        help='hidden dim protein'
    )
    parser.add_argument(
        '--cls_hidden_dims',
        type=list,
        default=[200, 100],
        help='cls hidden dims'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='batch size'
    )
    parser.add_argument(
        '--train_epoch',
        type=int,
        default=100,
        help='train epoch'
    )
    parser.add_argument(
        '--LR',
        type=float,
        default=0.005,
        help='learning rate'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=2,
        help='num workers'
    )
    parser.add_argument(
        '--attention',
        type=bool,
        default=True,
        help='attention'
    )
    parser.add_argument(
        '--mpnn_hidden_size',
        type=int,
        default=128,
        help='mpnn hidden size'
    )
    parser.add_argument(
        '--mpnn_depth',
        type=int,
        default=3,
        help='mpnn depth'
    )
    parser.add_argument(
        '--cnn_target_filters',
        type=list,
        default=[16,32,48],
        help='cnn target filters'
    )
    parser.add_argument(
        '--cnn_target_kernels',
        type=list,
        default=[24,48,72],
        help='cnn target kernels'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default="model",
        help='model path'
    )
    parser.add_argument(
        '--visual_attention',
        type=bool,
        default=False,
        help='visual attention'
    )
    parser.add_argument(
        '--concatenation',
        type=bool,
        default=False,
        help='concatenation'
    )
    flags, unparsed = parser.parse_known_args()
    return flags


def get_config():
    """
    set all configurations
    """
    flags=argparser()
    config = {}
    #the input and hidden dimension of drug/protein
    config['input_dim_drug'] = flags.input_dim_drug
    config['input_dim_protein'] = flags.input_dim_protein
    config['hidden_dim_drug'] = flags.hidden_dim_drug  # hidden dim of drug
    config['hidden_dim_protein'] = flags.hidden_dim_protein  # hidden dim of protein
    config['cls_hidden_dims'] = flags.cls_hidden_dims  # decoder classifier dim 1

    #training settings
    config['batch_size'] = flags.batch_size
    config['train_epoch'] = flags.train_epoch
    config['LR'] = flags.LR
    config['num_workers'] = flags.num_workers
    config['attention']=flags.attention

    #parameters of MPNN
    config['mpnn_hidden_size'] = flags.mpnn_hidden_size
    config['mpnn_depth'] = flags.mpnn_depth

    #parameters of CNN
    config['cnn_target_filters'] = flags.cnn_target_filters
    config['cnn_target_kernels'] = flags.cnn_target_kernels

    #with/without attention mechanism, with/without concatenation, model_save path
    config['modelpath'] = flags.model_path
    config['visual_attention']= flags.visual_attention
    config['concatenation']= flags.concatenation
    return config


def plot(train_epoch, acc_record, f1_record, precision_record, recall_record, loss_record, train_acc_record,
         train_f1_record, train_precision_record, train_recall_record, train_loss_record):
    # plot five statistic .png picture, including accuracy, F1 score, precision, recall, training loss
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

#save model parameters
def save_model(model, path_dir, **config):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    if torch.cuda.device_count()>1:
        torch.save(model.module.state_dict(), path_dir + '/model.pt')
    else:
        torch.save(model.state_dict(), path_dir + '/model.pt')
    save_dict(path_dir, config)

#load model parameters
def load_model(model, path_dir):
    para = torch.load(path_dir + '/model.pt')
    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(para)
    else:
        model.load_state_dict(para)
    return model


def test(device, data_generator, model):
    """
    testing on validation/test generator
    return prediction result, accuracy, precision, recall, F1
    """
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
    return pred_res, accuracy_score(true_y, predicted_y), precision_score(true_y, predicted_y), recall_score(true_y,predicted_y), f1_score(true_y, predicted_y)


def train(model, device, train_set, val_set, test_set, **config):
    """
    define the training process
    """
    #training parameters
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
            p = p.float().to(device)
            pred = model(d, p)
            label = Variable(torch.from_numpy(np.array(label)).long()).to(device)
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(pred, label)
            loss_val += loss.item() * label.size(0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()
        train_loss_record.append(loss_val/len(trainset_generator))
        tmp = time.time()

        # Output the training process
        print(' Epoch: ' + str(epo + 1) + '  Loss ' + str(loss_val) + ". Consumed Time " + str(
            int(tmp - start) / 60) + " mins", file=open(os.path.join(log_dir, 'log.txt'), 'a+'), flush=True)
        print(' Epoch: ' + str(epo + 1) + '  Loss ' + str(loss_val) + ". Consumed Time " + str(int(tmp - start) / 60) + " mins", flush=True)
        start = tmp

        # test current model every five epoches
        if (epo+1)%5==0:
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
            loss_record.append(lloss/len(validset_generator))
        save_model(model, log_dir, **config)

    #plot and print out the final evaluation result
    plot(train_epoch, acc_record, f1_record, precision_record, recall_record, loss_record, train_acc_record,
         train_f1_record, train_precision_record, train_recall_record, train_loss_record)
    pred_res, accuracy, precision, recall, f1 = test(device, testing_generator, model)
    print('Test at Epoch ' + str(epo + 1) + ' , Accuracy: ' + str(accuracy) + ', Precision:' + str(
            precision) + ' , Recall: ' + str(recall) + ' , F1: ' + str(f1),
            file=open(os.path.join(log_dir, 'log.txt'), 'a+'), flush=True)
    print('Test at Epoch ' + str(epo + 1) + ' , Accuracy: ' + str(accuracy) + ', Precision:' + str(
            precision) + ' , Recall: ' + str(recall) + ' , F1: ' + str(f1), flush=True)

    #save predictive outputs
    pred_res.to_csv(os.path.join(log_dir, 'predicted_labels.csv'))


def main():
    #read in dataset
    X_drugs, X_targets, y = read_file_training_dataset_drug_target_pairs('../train/train_new.csv')
    #data splition and dataloader generation
    train_set, val_set, test_set = data_process(X_drugs, X_targets, y, frac=[0.8, 0.1, 0.1], random_seed=2,drug_encoding="MPNN",target_encoding="CNN")
    #get the configuration of the model from parser
    config = get_config()
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    #define and initialize the model
    model = MPNN_CNN(**config)
    #train on the model and test on validation set
    train(model, train_device, train_set, val_set, test_set, **config)


if __name__ == '__main__':
    main()

