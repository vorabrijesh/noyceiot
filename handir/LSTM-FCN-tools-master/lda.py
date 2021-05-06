import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set() 
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("paper")
from pylab import rcParams 
rcParams['figure.figsize'] = 5, 5
from matplotlib.pyplot import cm
import argparse
import numpy as np
import pandas as pd
from scipy import interp
from itertools import cycle
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, classification_report
from utils.constants import MAX_SEQUENCE_LENGTH_LIST, NB_CLASSES_LIST



def lda(datasetindex, prefix, gen):

    """
    Training and running LDA

    datasetindex = Dataset index (number in constants.py)
    prefix = Prefix to dataset
    gen = Use generated synthetic data or not (0 or 1)
    """

    prefix = prefix
    DATASET_INDEX = datasetindex
    MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH_LIST[DATASET_INDEX]
    NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX]

    aa = pd.read_csv('../data/{prefix}_TEST'.format(prefix=prefix), header=None)
    len_aa = len(aa)
    y_test = aa.iloc[:,0]
    X_test = np.array(aa.iloc[:,1:])

    if gen == 0:
        aa = pd.read_csv('../data/{prefix}_TRAIN'.format(prefix=prefix), header=None)
    elif gen == 1:
        aa = pd.read_csv('../data/{prefix}_EXP_TRAIN'.format(prefix=prefix), header=None)


    len_aa = len(aa)
    y_train = aa.iloc[:,0]
    X_train = np.array(aa.iloc[:,1:])

    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X_train, y_train)
    y_score = lda.fit(X_train, y_train).decision_function(X_test)

    y_test = label_binarize(y_test, classes=range(0, NB_CLASS))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(NB_CLASS):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NB_CLASS)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NB_CLASS):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= NB_CLASS

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    read_dictionary = np.load('../data/{prefix}_labels_dict.npy'.format(prefix=prefix)).item()
    read_dictionary = {v: k for k, v in read_dictionary.items()}
    NB_CLASS_list = [read_dictionary[i] for i in range(NB_CLASS)]


    
    # Plot ROC

    plt.figure()
    plt.plot(fpr["micro"], 
             tpr["micro"],
             label='micro-average (AUC = {0:0.3f})'
             ''.format(roc_auc["micro"]), 
             color='deeppink', 
             linestyle=':', 
             linewidth=3)

    plt.plot(fpr["macro"], 
             tpr["macro"],
             label='macro-average (AUC = {0:0.3f})'
             ''.format(roc_auc["macro"]), 
             color='navy', 
             linestyle=':', 
             linewidth=3)

    colors=iter(cm.rainbow(np.linspace(0, 1, NB_CLASS)))
    for i, color in zip(range(NB_CLASS), colors):
        j = NB_CLASS_list[i]
        plt.plot(fpr[i], 
                 tpr[i], 
                 color=color, 
                 lw=3,
                 label='{0} (AUC = {1:0.3f})'
                 ''.format(j, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.025, 1.0])
    plt.ylim([0.0, 1.025])
    plt.rc('xtick',labelsize=16)
    plt.rc('ytick',labelsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(bbox_to_anchor=(1.1, 1.05), fontsize=16)
    plt.axis('scaled')
    # plt.show()

    plt.savefig('figures/{prefix}_lda_roc_curves.png'.format(prefix=prefix), bbox_inches='tight')


    print("""
          {prefix},{model_num},{micro},{macro}
          """.format(prefix=prefix,
                     model_num='LDA',
                     micro=roc_auc["micro"],
                     macro=roc_auc["macro"]),
          file=open('weights/ROCAUCs.csv', "a"))


    # Save micro and macro ROC curves dataframe to CSV
    micro = pd.DataFrame(np.vstack((fpr["micro"], tpr["micro"]))).transpose()
    macro = pd.DataFrame(np.vstack((fpr["macro"], tpr["macro"]))).transpose()
    micro_macro = pd.concat([micro, macro], axis=1)
    micro_macro.columns = ['fpr_micro', 'tpr_micro', 'fpr_macro', 'tpr_macro']
    micro_macro.to_csv('weights/{prefix}_{model_num}_micro_macro.csv'.format(model_num='LDA', prefix=prefix))



    # Classifcation_report
    ## print(classification_report(y_true, y_pred, target_names=target_names))
    print(classification_report(np.argmax(y_test, axis=1), lda.predict(X_test), 
            target_names=[l for l in read_dictionary.values()]),
    file=open('weights/{prefix}_{model_num}_report.txt'.format(model_num='LDA', prefix=prefix), "w"))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datasetindex', help='Dataset index', type=int, required=True)
    parser.add_argument('-p', '--prefix', help='Prefix to dataset', type=str, default='Output')
    parser.add_argument('-g', '--gen', help='Use generated synthetic data or not (0 or 1)', type=int, default=0)
    args = parser.parse_args()

    if not args.datasetindex:
        parser.error('No dataset index given, add --datasetindex')

    lda(args.datasetindex, args.prefix, args.gen)