from torch.utils.data import SequentialSampler
import time
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score, f1_score, log_loss, \
    accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from models import *

# seed setting
torch.manual_seed(1)
np.random.seed(1)

log_dir = os.path.join('log', time.asctime(time.localtime(
    time.time()))).replace(" ", "_").replace(":", "_")


def model_initialize(**config):
    model = MPNN_CNN(**config)
    return model


class MPNN_CNN:
    def __init__(self, **config):
        #  set the models for drugs and targets respectively
        #  Drug: MPNN  Target:CNN
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.model_drug = MPNN(config['hidden_dim_drug'], config['mpnn_depth'])
        self.model_protein = CNN('protein', **config)

        self.model = Classifier(self.model_drug, self.model_protein, **config)
        self.config = config
        self.device = device
        self.drug_encode_method = 'MPNN'
        self.target_encode_method = 'CNN'
        self.store_url = config['result_folder']
        if not os.path.exists(self.store_url):
            os.mkdir(self.store_url)
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

    def test_(self, data_generator, model):
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
        return pred_res, accuracy_score(true_y, predicted_y), precision_score(true_y, predicted_y), recall_score(true_y,
                                                                                                                 predicted_y), f1_score(
            true_y, predicted_y)

    def train(self, train, val=None, test=None):
        lr = self.config['LR']
        BATCH_SIZE = self.config['batch_size']
        train_epoch = self.config['train_epoch']
        self.model = self.model.to(self.device)
        # split the workload to all cuda evenly
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model, dim=0)

        # dynamically change the lr
        # for every 5 epoches  lr:=lr*0.8
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        miles = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
                 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=miles, gamma=0.8)

        # data loader for train val and test(if val and test existes)
        params = {'batch_size': BATCH_SIZE, 'shuffle': True,
                  'num_workers': self.config['num_workers'], 'drop_last': False}
        params['collate_fn'] = mpnn_collate_func
        trainset_generator = data.DataLoader(data_process_loader(
            train.index.values, train.Label.values, train, **self.config), **params)
        if val is not None:
            validset_generator = data.DataLoader(data_process_loader(
                val.index.values, val.Label.values, val, **self.config), **params)
        if test is not None:
            info = data_process_loader(
                test.index.values, test.Label.values, test, **self.config)
            params_test = {'batch_size': BATCH_SIZE, 'shuffle': False,
                           'num_workers': self.config['num_workers'], 'drop_last': False,
                           'sampler': SequentialSampler(info)}
            params_test['collate_fn'] = params['collate_fn']
            testing_generator = data.DataLoader(data_process_loader(
                test.index.values, test.Label.values, test, **self.config), **params_test)

        # recode the metrics when training
        acc_record, f1_record, precision_record, recall_record, loss_record = [], [], [], [], []

        t_prev = time.time()
        for epo in range(train_epoch):
            loss_val = 0
            for i, (d, p, label) in enumerate(trainset_generator):
                p = p.float().to(self.device)
                pred = self.model(d, p)
                label = Variable(torch.from_numpy(
                    np.array(label)).long()).to(self.device)
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(pred, label)
                loss_val += loss.item() * label.size(0)
                opt.zero_grad()
                loss.backward()
                opt.step()
                lr_scheduler.step()
            t_now = time.time()
            # Output the training process
            print(' Epoch: ' + str(epo + 1) +
                  '  Loss ' + str(loss_val)[:7] +
                  ". Consumed Time " + str(int(t_now - t_prev) / 3600)[:7] + " hours",
                  file=open(os.path.join(log_dir, 'log.txt'), 'a+'), flush=True)
            print(' Epoch: ' + str(epo + 1) +
                  '  Loss ' + str(loss_val)[:7] +
                  ". Consumed Time " + str(int(t_now - t_prev) / 3600)[:7] + " hours", flush=True)
            t_prev = t_now

            with torch.set_grad_enabled(False):
                _,accuracy, precision, recall, f1 = self.test_(
                    trainset_generator, self.model)
                print('Training at Epoch ' + str(epo + 1) + ', Accuracy: ' + str(accuracy)[:7] + ', Precision: ' + str(
                    precision)[:7]
                      + ', Recall: ' + str(recall)[:7] + ' , F1: ' + str(f1)[:7], file=open(log_dir + 'log.txt', 'a+'),
                      flush=True)
                print('Training at Epoch ' + str(epo + 1) + ', Accuracy: ' + str(accuracy)[:7] + ', Precision: ' + str(
                    precision)[:7]
                      + ', Recall: ' + str(recall)[:7] + ' , F1: ' + str(f1)[:7], flush=True)
            if val is not None:
                with torch.set_grad_enabled(False):
                    _,accuracy, precision, recall, f1 = self.test_(
                        validset_generator, self.model)
                    print('Validation at Epoch ' + str(epo + 1) + ', Accuracy: ' + str(accuracy)[
                                                                                   :7] + ', Precision: ' + str(
                        precision)[:7]
                          + ', Recall: ' + str(recall)[:7] + ' , F1: ' + str(f1)[:7],
                          file=open(log_dir + 'log.txt', 'a+'), flush=True)
                    print('Validation at Epoch ' + str(epo + 1) + ', Accuracy: ' + str(accuracy)[
                                                                                   :7] + ', Precision: ' + str(
                        precision)[:7]
                          + ', Recall: ' + str(recall)[:7] + ' , F1: ' + str(f1)[:7], flush=True)
                    acc_record.append(accuracy)
                    f1_record.append(f1)
                    precision_record.append(precision)
                    recall_record.append(recall)
                    lloss = 0
                    for i, (d, p, label) in enumerate(validset_generator):
                        p = p.float().to(self.device)
                        pred = self.model(d, p)
                        label = Variable(torch.from_numpy(
                            np.array(label)).long()).to(self.device)
                        loss_fct = torch.nn.CrossEntropyLoss()
                        loss_ = loss_fct(pred, label)
                        lloss += loss_.item() * label.size(0)
                    loss_record.append(lloss)
        self.plot(train_epoch, acc_record, f1_record, precision, recall_record, loss_record)

        pred_res = []
        if test is not None:
            pred_res, accuracy, precision, recall, f1 = self.test_(
                testing_generator, self.model)
            print(
                'Test at Epoch ' + str(epo + 1) + ' , Accuracy: ' + str(accuracy)[:7] + ', Precision:' + str(precision)[
                                                                                                         :7]
                + ' , Recall: ' + str(recall)[:7] + ' , F1: ' + str(f1)[:7],
                file=open(os.path.join(log_dir, 'log.txt'), 'a+'), flush=True)
            print(
                'Test at Epoch ' + str(epo + 1) + ' , Accuracy: ' + str(accuracy)[:7] + ', Precision:' + str(precision)[
                                                                                                         :7]
                + ' , Recall: ' + str(recall)[:7] + ' , F1: ' + str(f1)[:7], flush=True)
        pred_res.to_csv(os.path.join(log_dir, 'predicted_labels.csv'))

        self.save_model(log_dir)

    def save_model(self, path_dir):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        torch.save(self.model.state_dict(), path_dir + '/model.pt')
        save_dict(path_dir, self.config)
