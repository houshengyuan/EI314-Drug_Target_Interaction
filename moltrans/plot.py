
import matplotlib.pyplot as plt
import os
import pickle


def plot_loss(history,filepath):
   plt.figure()
   plt.plot(history['loss'])
   plt.plot(history['val_loss'])
   plt.title('model loss')
   plt.ylabel('loss')
   plt.xlabel('epoch')
   plt.legend(['trainloss', 'valloss'], loc='upper left')
   plt.savefig(os.path.join(filepath,"loss.png"), dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
             format=None, transparent=False, bbox_inches=None, pad_inches=0.1)
   plt.close()

   plt.figure()
   plt.title('model accuracy')
   plt.ylabel('accuracy')
   plt.xlabel('epoch')
   plt.ylim((0, 1))
   plt.plot(history['binary_accuracy'])
   plt.plot(history['val_binary_accuracy'])
   plt.legend(['train_acc', 'val_acc'], loc='upper left')
   plt.savefig(os.path.join(filepath,"acc.png"), dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
               format=None, transparent=False, bbox_inches=None, pad_inches=0.1)
   plt.close()

   plt.figure()
   plt.title('model F1')
   plt.ylabel('F1')
   plt.xlabel('epoch')
   plt.ylim((0, 1))
   plt.plot(history['f1_score'])
   plt.plot(history['val_f1_score'])
   plt.legend(['train_f1', 'val_f1'], loc='upper left')
   plt.savefig(os.path.join(filepath,"f1.png"), dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
               format=None, transparent=False, bbox_inches=None, pad_inches=0.1)
   plt.close()

   plt.figure()
   plt.title('model precision')
   plt.ylabel('Precision')
   plt.xlabel('epoch')
   plt.ylim((0, 1))
   plt.plot(history['precision'])
   plt.plot(history['val_precision'])
   plt.legend(['train_precision', 'val_precision'], loc='upper left')
   plt.savefig(os.path.join(filepath,"precision.png"), dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
               format=None, transparent=False, bbox_inches=None, pad_inches=0.1)
   plt.close()

   plt.figure()
   plt.title('model recall')
   plt.ylabel('Recall')
   plt.xlabel('epoch')
   plt.ylim((0, 1))
   plt.plot(history['recall'])
   plt.plot(history['val_recall'])
   plt.legend(['train_recall', 'val_recall'], loc='upper left')
   plt.savefig(os.path.join(filepath,"recall.png"), dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
                format=None, transparent=False, bbox_inches=None, pad_inches=0.1)
   plt.close()

if __name__=="__main__":
 model_dir="log/Tue_May__4_12_25_33_2021"
 model_result=pickle.load(open(os.path.join(model_dir,"result.pkl"),"rb"))
 plot_loss(model_result,model_dir)
