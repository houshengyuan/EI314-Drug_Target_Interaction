from model import MolTrans
from config import CONFIG
from encode import drug_encoder, protein_encoder
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.callbacks import History
from tensorflow.keras.layers import Dense
from plot import plot_loss
from keras.utils.vis_utils import plot_model
import os
import time
import numpy as np
import pandas as pd
import random
import keras_metrics
from argparse import ArgumentParser
import json
import pickle

np.random.seed(0)
random.seed(0)
print('GPU:', tf.test.is_gpu_available())
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

tf.set_random_seed(0)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
history = History()

log_dir = os.path.join('log',
                       time.asctime(time.localtime(time.time())).replace(" ", "_").replace(":", "_"))
filepath = os.path.join(log_dir, "log.txt")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = ArgumentParser(description='MolTrans Program.')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16)', dest='batch_size')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs', dest='epochs')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')


def load_data(path, type='train'):
    data = pd.read_csv(path, encoding='utf-8')
    smiles = np.array(data['SMILES'])
    fasta = np.array(data['FASTA'])
    label = np.array(data['Label'])

    drugcount = len(set(data['SMILES'].tolist()))
    targetcount = len(set(data['FASTA'].tolist()))
    if type == "train":
        print("train drugs: ", drugcount)
        print("train targets: ", targetcount)
    elif type == "val":
        print("val drugs: ", drugcount)
        print("val targets: ", targetcount)
    return smiles, fasta, label


def encode_data(drug, protein, type='train'):
    drug_v = []
    drug_mask = []
    for i in drug:
        v, mask = drug_encoder(i)
        drug_mask.append(mask)
        drug_v.append(v)
    protein_mask = []
    protein_v = []
    for i in protein:
        v, mask = protein_encoder(i)
        protein_mask.append(mask)
        protein_v.append(v)
    drug_mask = np.array(drug_mask)
    drug_v = np.array(drug_v)
    protein_mask = np.array(protein_mask)
    protein_v = np.array(protein_v)
    if type == "train":
        print("train drug mask shape: ", drug_mask.shape)
        print("train protein mask shape: ", protein_mask.shape)
    elif type == "val":
        print("validation drug mask shape: ", drug_mask.shape)
        print("validation protein mask shape: ", protein_mask.shape)
    return drug_v, protein_v, drug_mask, protein_mask


def main():
    config = CONFIG()
    args = parser.parse_args()
    config['batch_size']=args.batch_size
    #print(config['batch_size'])
    optim = keras.optimizers.Adam(args.lr)

    print("Encoding training data...")
    train_drug, train_protein, train_label = load_data(os.path.join('../train', 'train.csv'), type='train')
    train_drug_v, train_protein_v, train_drug_mask, train_protein_mask = encode_data(train_drug, train_protein,
                                                                                     type='train')
    print("Finish Encoding training data.")
    #print(train_drug_v.shape)
    #print(train_protein_v.shape)
    #exit(1)

    print("Encoding validation data...")
    val_drug, val_protein, val_label = load_data(os.path.join('../validation', 'validation.csv'), type='val')
    val_drug_v, val_protein_v, val_drug_mask, val_protein_mask = encode_data(val_drug, val_protein, type='val')
    print("Finish Encoding validation data.")

    inputs1 = keras.Input(shape=(50,))
    inputs2 = keras.Input(shape=(545,))
    inputs3 = keras.Input(shape=(50,))
    inputs4 = keras.Input(shape=(545,))
    moltrans = MolTrans(**config)(inputs1, inputs2, inputs3, inputs4)
    moltrans_model = keras.Model(inputs=[inputs1, inputs2, inputs3, inputs4], outputs=[moltrans])
    moltrans_model.compile(optimizer=optim, loss='binary_crossentropy', metrics=[keras.metrics.binary_accuracy,
                                                                                 keras_metrics.binary_precision(),
                                                                                 keras_metrics.binary_recall(),
                                                                                 keras_metrics.f1_score(),
                                                                                 keras_metrics.true_positive(),
                                                                                 keras_metrics.true_negative(),
                                                                                 keras_metrics.false_positive(),
                                                                                 keras_metrics.false_negative()
                                                                                 ])
    # model=model.cuda()
    print(moltrans_model.summary(), file=open(filepath, 'a+'), flush=True)

    #plot_model(moltrans_model, to_file=os.path.join(log_dir, 'build_combined_categorical.png'))
    print(train_drug_v.shape)
    print(train_drug_mask.shape)
    print(train_protein_v.shape)
    print(train_protein_mask.shape)
    sess.run(tf.initialize_all_variables())
    sess.run(tf.global_variables_initializer())
    keras.backend.get_session().run(tf.global_variables_initializer())
    res = moltrans_model.fit(x=[train_drug_v, train_protein_v, train_drug_mask, train_protein_mask], y=train_label,
                             batch_size=args.batch_size, epochs=args.epochs, shuffle=True,
                             validation_data=([val_drug_v, val_protein_v, val_drug_mask, val_protein_mask], val_label),
                             verbose=2)
    predict_label=moltrans_model.predict([val_drug_v, val_protein_v, val_drug_mask, val_protein_mask])

    json.dump(predict_label.tolist(), open(os.path.join(log_dir, "predicted_labels.txt"), "w+"))
    trainperf = moltrans_model.evaluate(([train_drug_v, train_protein_v, train_drug_mask, train_protein_mask]), train_label, verbose=0)
    valperf = moltrans_model.evaluate(([val_drug_v, val_protein_v, val_drug_mask, val_protein_mask]), val_label, verbose=2)
    pickle.dump(res.history, open(os.path.join(log_dir, "result.pkl"), "wb+"))
    plot_loss(res.history, log_dir)
    moltrans_model.save_weights(os.path.join(log_dir, "model_weights.h5"))

    print("---FINAL RESULTS-----", file=open(filepath, 'a+'), flush=True)
    print("Validation Performance ", file=open(filepath, 'a+'), flush=True)
    metrics = ['Loss', 'accuracy', 'precision', 'recall', 'F1', 'TP', 'TN', 'FP', 'FN']
    for i in range(len(valperf)):
        print(metrics[i] + ": " + str(valperf[i]), file=open(filepath, 'a+'), flush=True)
    print("Train Performance ", file=open(filepath, 'a+'), flush=True)
    for i in range(len(trainperf)):
        print(metrics[i] + ": " + str(trainperf[i]), file=open(filepath, 'a+'), flush=True)

if __name__ == "__main__":
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print(CONFIG(), file=open(filepath, 'a+'), flush=True)
    main()
