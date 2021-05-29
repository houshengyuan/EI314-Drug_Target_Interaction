from torch.utils.data import SequentialSampler
import time
from sklearn.metrics import  f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from models import *
import os

# seed setting
torch.manual_seed(1)
np.random.seed(1)
log_dir = os.path.join('log', time.asctime(time.localtime(time.time()))).replace(" ", "_").replace(":", "_")



class MPNN_CNN:
    def __init__(self, **config):
        #  set the models for drugs and targets respectively
        #  Drug: MPNN  Target:CNN
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if self.config["preTrain"]==True:
            self.load_model(self.config['modelpath'])
        else:
            self.model = Classifier(self.model_drug, self.model_protein, **config)
        self.model_drug = MPNN(self.config['hidden_dim_drug'], self.config['mpnn_depth'])
        self.model_protein = CNN('protein', **config)
        self.device = device
        self.drug_encode_method = 'MPNN'
        self.target_encode_method = 'CNN'
        self.lr=self.config['LR']
        self.batch_size=self.config['batch_size']
        self.train_epoch=self.config['train_epoch']
        self.num_workers=self.config['num_workers']
        if 'num_workers' not in self.config.keys():
            self.config['num_workers'] = 0

    def plot(self, train_epoch, acc_record, f1_record, precision_record, recall_record, loss_record):
        # plot five statistic .png picture
        plt.plot(np.arange(1, train_epoch + 1), np.array(acc_record))
        plt.savefig(os.path.join(log_dir, 'acc.png'))
        plt.clf()
        plt.plot(np.arange(1, train_epoch + 1), np.array(f1_record))
        plt.savefig(os.path.join(log_dir, 'f1.png'))
        plt.clf()
        plt.plot(np.arange(1, train_epoch + 1), np.array(precision_record))
        plt.savefig(os.path.join(log_dir, 'precision.png'))
        plt.clf()
        plt.plot(np.arange(1, train_epoch + 1), np.array(recall_record))
        plt.savefig(os.path.join(log_dir, 'recall.png'))
        plt.clf()
        plt.plot(np.arange(1, train_epoch + 1), np.array(loss_record))
        plt.savefig(os.path.join(log_dir, 'loss.png'))
        plt.clf()

    def test(self, data_generator, model):
        predicted_y = []
        true_y = []
        with torch.no_grad():
            model.eval()
            for i, (d, p, label) in enumerate(data_generator):
                p = p.float().to(self.device)
                pred = self.model(d, p)
                predictions = torch.max(pred.data, 1)[
                    1].detach().cpu().numpy()
                label_ids = label.to('cpu').numpy()
                true_y += label_ids.flatten().tolist()
                predicted_y += predictions.flatten().tolist()
            pred_res = pd.DataFrame(predicted_y)
        model.train()
        return pred_res, accuracy_score(true_y, predicted_y), precision_score(true_y, predicted_y), recall_score(true_y,predicted_y), f1_score(true_y, predicted_y)

    def train(self, train, val=None, test=None):
        self.model = self.model.to(self.device)
        # split the workload to all cuda evenly
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model, dim=0)
        # dynamically change the lr
        # for every 5 epoches  lr:=lr*0.8
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        miles = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
                 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=miles, gamma=0.8)
        # data loader for train val and test(if val and test existes)
        params = {'batch_size': self.batch_size, 'shuffle': True,'num_workers': self.config['num_workers'], 'drop_last': False}
        trainset_generator = data.DataLoader(data_loader(
            train.index.values, train.Label.values, train, **self.config), **params)
        validset_generator = data.DataLoader(data_loader(
                val.index.values, val.Label.values, val, **self.config), **params)
        info = data_loader(test.index.values, test.Label.values, test, **self.config)
        params_test = {'batch_size': self.batch_size, 'shuffle': False,
                           'num_workers': self.config['num_workers'], 'drop_last': False,
                           'sampler': SequentialSampler(info)}
        testing_generator = data.DataLoader(data_loader(test.index.values, test.Label.values, test, **self.config), **params_test)
        # recode the metrics when training
        acc_record, f1_record, precision_record, recall_record, loss_record = [], [], [], [], []
        start = time.time()
        for epo in range(self.train_epoch):
            loss_val = 0
            for i, (d, p, label) in enumerate(trainset_generator):
                p = p.float().to(self.device)
                pred = self.model(d, p)
                label = Variable(torch.from_numpy(np.array(label)).long()).to(self.device)
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(pred, label)
                loss_val += loss.item() * label.size(0)
                opt.zero_grad()
                loss.backward()
                opt.step()
                lr_scheduler.step()
            tmp = time.time()
            # Output the training process
            print(' Epoch: ' + str(epo + 1) +'  Loss ' + str(loss_val) +". Consumed Time " + str(int(tmp - start) / 60) + " mins",file=open(os.path.join(log_dir, 'log.txt'), 'a+'), flush=True)
            print(' Epoch: ' + str(epo + 1) +'  Loss ' + str(loss_val) +". Consumed Time " + str(int(tmp - start) / 60) + " mins", flush=True)
            start = tmp
            with torch.set_grad_enabled(False):
                _,accuracy, precision, recall, f1 = self.test(trainset_generator, self.model)
                print('Training at Epoch ' + str(epo + 1) + ', Accuracy: ' + str(accuracy) + ', Precision: ' + str(
                    precision)+ ', Recall: ' + str(recall) + ' , F1: ' + str(f1), file=open(log_dir + 'log.txt', 'a+'),flush=True)
                print('Training at Epoch ' + str(epo + 1) + ', Accuracy: ' + str(accuracy)+ ', Precision: ' + str(
                    precision)+ ', Recall: ' + str(recall) + ' , F1: ' + str(f1), flush=True)
                _,accuracy, precision, recall, f1 = self.test(validset_generator, self.model)
                print('Validation at Epoch ' + str(epo + 1) + ', Accuracy: ' + str(accuracy)+ ', Precision: ' + str(precision)
                          + ', Recall: ' + str(recall) + ' , F1: ' + str(f1),file=open(log_dir + 'log.txt', 'a+'), flush=True)
                print('Validation at Epoch ' + str(epo + 1) + ', Accuracy: ' + str(accuracy)+ ', Precision: ' + str(
                        precision)+ ', Recall: ' + str(recall) + ' , F1: ' + str(f1), flush=True)
                acc_record.append(accuracy)
                f1_record.append(f1)
                precision_record.append(precision)
                recall_record.append(recall)
                lloss = 0
                for i, (d, p, label) in enumerate(validset_generator):
                    p = p.float().to(self.device)
                    pred = self.model(d, p)
                    label = Variable(torch.from_numpy(np.array(label)).long()).to(self.device)
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss_ = loss_fct(pred, label)
                    lloss += loss_.item() * label.size(0)
                loss_record.append(lloss)
        self.plot(self.train_epoch, acc_record, f1_record, precision, recall_record, loss_record)
        pred_res, accuracy, precision, recall, f1 = self.test(testing_generator, self.model)
        print('Test at Epoch ' + str(epo + 1) + ' , Accuracy: ' + str(accuracy)+ ', Precision:' + str(precision)+ ' , Recall: ' + str(recall) + ' , F1: ' + str(f1),file=open(os.path.join(log_dir, 'log.txt'), 'a+'), flush=True)
        print('Test at Epoch ' + str(epo + 1) + ' , Accuracy: ' + str(accuracy) + ', Precision:' + str(precision)+ ' , Recall: ' + str(recall) + ' , F1: ' + str(f1), flush=True)
        pred_res.to_csv(os.path.join(log_dir, 'predicted_labels.csv'))
        self.save_model(log_dir)

    def save_model(self, path_dir):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        torch.save(self.model.state_dict(), path_dir + '/model.pt')
        save_dict(path_dir, self.config)

    def load_model(self, path_dir):
        para=torch.load(path_dir + '/model.pt')
        self.model.load_state_dict(para)
        self.config=load_dict(path_dir)
