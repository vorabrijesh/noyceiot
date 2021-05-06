import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
sns.set() 
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("paper")
from pylab import rcParams, interp
import keras
rcParams['figure.figsize'] = 5, 5

from utils.constants import MAX_SEQUENCE_LENGTH_LIST, NB_CLASSES_LIST
from utils.keras_utils   import evaluate_model, set_trainable, visualize_context_vector, visualize_cam
import argparse
import pandas as pd
import numpy as np

from keras.models import Model
from keras.layers import Input, PReLU, Dense, LSTM, multiply, concatenate, Activation
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from utils.constants import MAX_SEQUENCE_LENGTH_LIST, NB_CLASSES_LIST
from utils.keras_utils import train_model
from utils.layer_utils import AttentionLSTM
from keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
import scipy
from matplotlib.pyplot import cm
import datetime



def trainer(datasetindex, model_num, run_training, prefix, epochs, batch_size, lstm_cell_num):

    """
    Training the model

    datasetindex = Dataset index (number in constants.py)
    run_training = Run training, if run_training = 1
    prefix = Prefix to dataset
    epochs = Number of epochs for training
    batch_size = Batch size for training
    lstm_cell_num = LSTM cell number 
    """


    DATASET_INDEX = datasetindex # 1
    MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH_LIST[DATASET_INDEX]
    NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX]


    # Allow for different default LCN settings for each LSTM-containing model
    if (model_num == 0 and lstm_cell_num == None):
        lstm_cell_num = 128
    elif (model_num == 2 and lstm_cell_num == None):
        lstm_cell_num = 32
    elif (model_num == 3 and lstm_cell_num == None):
        lstm_cell_num = 32


    # Allow for different default batch size settings for each model
    if (model_num == 0 and batch_size == None):
        batch_size = 128
    elif (model_num == 1 and batch_size == None):
        batch_size = 4
    elif (model_num == 2 and batch_size == None):
        batch_size = 4
    elif (model_num == 3 and batch_size == None):
        batch_size = 4


    # LSTM
    def generate_model_0():
        ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))
        x = LSTM(lstm_cell_num)(ip)
        # x = Dropout(0.8)(x)
        out = Dense(NB_CLASS, activation='softmax')(x)
        model = Model(ip, out)

        return model


    # FCN
    def generate_model_1():
        ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))
        y = Permute((2, 1))(ip)
        y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = GlobalAveragePooling1D()(y)
        out = Dense(NB_CLASS, activation='softmax')(y)
        model = Model(ip, out)

        return model


    # LSTM-FCN
    def generate_model_2():
        ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))
        x = LSTM(lstm_cell_num)(ip)   # 8
        x = Dropout(0.8)(x)
        y = Permute((2, 1))(ip)
        y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = GlobalAveragePooling1D()(y)
        x = concatenate([x, y])
        out = Dense(NB_CLASS, activation='softmax')(x)
        model = Model(ip, out)

        return model


    # ALSTM-FCN
    def generate_model_3():
        ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))
        x = AttentionLSTM(lstm_cell_num)(ip) # 8
        x = Dropout(0.8)(x)
        y = Permute((2, 1))(ip)
        y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = GlobalAveragePooling1D()(y)
        x = concatenate([x, y])
        out = Dense(NB_CLASS, activation='softmax')(x)
        model = Model(ip, out)

        return model


    # Select which model

    if model_num == 0:
        model = generate_model_0()
    elif model_num == 1:
        model = generate_model_1()
    elif model_num == 2:
        model = generate_model_2()
    elif model_num == 3:
        model = generate_model_3()


    # Run training or not ############################

    if run_training == 1: # -r
        history = train_model(model, 
                              DATASET_INDEX, # 1
                              dataset_prefix=prefix, # ArrowHead
                              epochs=epochs, # 500
                              batch_size=batch_size,
                              model_num=model_num) # 32
        hist = pd.DataFrame(model.history.history)
        hist.to_csv('./weights/{prefix}_{model_num}_history.csv'.format(prefix=prefix, model_num=model_num))



    # Plot history ####################################

    hist = pd.read_csv('./weights/{prefix}_{model_num}_history.csv'.format(prefix=prefix, model_num=model_num))
    hist.index = hist.index + 1
    getcol = ('acc', 'loss')
    r = 2; c = 1
    fig, ax = plt.subplots(r, c, sharex='col', sharey='row')
    for i in range(r):
        ax[i].plot(hist.filter(regex=getcol[i]))
        ax[i].legend(hist.filter(regex=getcol[i]).columns, fontsize=10)
        ax[i].set_ylabel(getcol[i], fontsize=18)
        ax[-1].set_xlabel('epochs', fontsize=18)
        ax[i].xaxis.set_tick_params(labelsize=10)
        ax[i].yaxis.set_tick_params(labelsize=10)


    plt.savefig('figures/{prefix}_{model_num}_training_history.png'.format(prefix=prefix, model_num=model_num), bbox_inches='tight')



    # Load weights if not training

    if run_training == 0: # -r
        optm = Adam(lr=1e-3)
        model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])
        model.load_weights('weights/{prefix}_{model_num}_weights.h5'.format(prefix=prefix, model_num=model_num))


    # Evaluate model ####################################

    evaluate_model(model, 
                   DATASET_INDEX, 
                   dataset_prefix=prefix, 
                   batch_size=batch_size,
                   model_num=model_num)


    # Visualize datasets ###############################
    ## Usable if Attention module for LSTM is included (ALSTM-FCN = model 3)

    if model_num >= 3:
        visualize_context_vector(model,
                                 DATASET_INDEX, 
                                 dataset_prefix=prefix,
                                 visualize_sequence=True,
                                 visualize_classwise=True, 
                                 limit=1,
                                 model_num=model_num)

        plt.savefig('figures/{prefix}_{model_num}_visualize_context_vector.png'.format(prefix=prefix, model_num=model_num), bbox_inches='tight')




    # Visualize what the convnet is learning ###########
    ## Usable if FCN is included (FCN = model 1, LSTM-FCN = model 2, ALSTM-FCN = model 3)
    

    if model_num >= 1:
        for j in range(NB_CLASS):
            a, b = visualize_cam(model, 
                          DATASET_INDEX, 
                          dataset_prefix=prefix, 
                          class_id=j, 
                          model_num=model_num)
            plt.savefig('figures/{prefix}_{model_num}_{class_num}_visualize_cam.png'.format(prefix=prefix, 
                                                                                model_num=model_num,
                                                                                class_num=j),
                                                                                bbox_inches='tight')
            a.to_csv('cams/{prefix}/{prefix}_{model_num}_{class_num}_cam_1.csv'.format(model_num=model_num, 
                                                                               prefix=prefix,
                                                                                class_num=j))
            b.to_csv('cams/{prefix}/{prefix}_{model_num}_{class_num}_cam_2.csv'.format(model_num=model_num, 
                                                                               prefix=prefix,
                                                                                class_num=j))



    read_dictionary = np.load('./data/{prefix}_labels_dict.npy'.format(prefix=prefix),allow_pickle=True).item()
    read_dictionary = {v: k for k, v in read_dictionary.items()}



    # Test individual samples

    aa = pd.read_csv('./data/{prefix}_TEST'.format(prefix=prefix), header=None)
    aa = aa.iloc[0]
    aa = np.delete(np.array(aa), 0)
    aa = aa.reshape(1, 1, MAX_SEQUENCE_LENGTH)
    aa_df = pd.DataFrame(np.transpose(model.predict(aa)))
    aa_df['Label'] = [read_dictionary[int(x)] for x in aa_df.index]
    aa_df.columns.values[0] = 'Probability' 
    aa_df.set_index('Label', inplace=True)
    ax = aa_df.sort_values('Probability').plot(kind='barh', colormap='RdBu', fontsize=16)
    ax.set_xlabel('Probability', fontsize=18)
    ax.set_ylabel('Label', fontsize=18)
    ax.legend().set_visible(False)

    plt.savefig('figures/{prefix}_{model_num}_test_sample.png'.format(prefix=prefix, model_num=model_num), bbox_inches='tight')



    # All test samples - confusion matrix
    
    aa = pd.read_csv('./data/{prefix}_TEST'.format(prefix=prefix), header=None)
    len_aa = len(aa)
    aa_labels = aa.iloc[:,0]
    aa = np.array(aa.iloc[:,1:])
    aa = aa.reshape(len_aa, 1, MAX_SEQUENCE_LENGTH)
    aa_pred = np.argmax(model.predict(aa), axis=1)
    aa_true = aa_labels
    aa_pred = pd.DataFrame([aa_pred]).T
    aa_both = pd.concat([aa_true, aa_pred], axis=1)
    aa_both = aa_both.replace(read_dictionary)
    aa_both_ct = pd.crosstab(aa_both.iloc[:,0], aa_both.iloc[:,1], 
                            normalize='index', rownames=['True'], colnames=['Predicted']).round(2)
    aa_both_ct.to_csv('weights/{prefix}_{model_num}_conf_mat.csv'.format(model_num=model_num, prefix=prefix))
    
    ## plot aa_both_ct
    sns.set(font_scale=0.8) #for label size # 1.4
    plt.clf()
    ax = sns.heatmap(aa_both_ct, annot=True, annot_kws={"size": 6}, cmap="coolwarm", vmin=0, vmax=1) # 10

    plt.savefig('figures/{prefix}_{model_num}_confusion_matrix.png'.format(prefix=prefix, model_num=model_num), bbox_inches='tight')






    # ROC

    aa = pd.read_csv('./data/{prefix}_EXP1_TEST'.format(prefix=prefix), header=None)
    len_aa = len(aa)
    aa_labels = aa.iloc[:,0]
    y_test = label_binarize(aa_labels, classes=range(0, NB_CLASS))
    aa = np.array(aa.iloc[:,1:])
    aa = aa.reshape(len_aa, 1, MAX_SEQUENCE_LENGTH)
    y_score = model.predict(aa)
    #print(y_test)
   # print(y_score)


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

    read_dictionary = np.load('./data/{prefix}_labels_dict.npy'.format(prefix=prefix),allow_pickle=True).item()
    read_dictionary = {v: k for k, v in read_dictionary.items()}
    NB_CLASS_list = [read_dictionary[i] for i in range(NB_CLASS)]

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

    plt.savefig('figures/{prefix}_{model_num}_roc_curves.png'.format(prefix=prefix, model_num=model_num), bbox_inches='tight')



    # Save micro and macro ROC curves dataframe to CSV
    micro = pd.DataFrame(np.vstack((fpr["micro"], tpr["micro"]))).transpose()
    macro = pd.DataFrame(np.vstack((fpr["macro"], tpr["macro"]))).transpose()
    micro_macro = pd.concat([micro, macro], axis=1)
    micro_macro.columns = ['fpr_micro', 'tpr_micro', 'fpr_macro', 'tpr_macro']
    micro_macro.to_csv('weights/{prefix}_{model_num}_micro_macro.csv'.format(model_num=model_num, prefix=prefix))



    ## Print micro- and macro-averages ROCAUC
    ### dataset,model,micro,macro
    
    print("""
          {prefix},{model_num},{micro},{macro}
          """.format(prefix=prefix,
                     model_num=model_num,
                     micro=roc_auc["micro"],
                     macro=roc_auc["macro"]),
          file=open('weights/ROCAUCs.csv', "a"))




    # Print out some parameters of the training run that's being run

    ts = str(datetime.datetime.now()).split('.')[0]
    print("""
          The dataset index {datasetindex} is for the dataset {prefix}. 
          Model number: {model_num}. Where LSTM = 0, FCN = 1, LSTM-FCN = 2, ALSTM-FCN = 3.
          The model was run for {epochs} epochs with a batch size of {batch_size}.
          Run at {ts}.
            """.format(datasetindex=datasetindex,
                         prefix=prefix, 
                         model_num=model_num,
                         epochs=epochs,
                         batch_size=batch_size,
                         ts=ts),
          file=open('weights/{prefix}_{model_num}_parameters.txt'.format(prefix=prefix, model_num=model_num), "w"))





    # Classifcation_report
    ## print(classification_report(y_true, y_pred, target_names=target_names))
    aa = pd.read_csv('./data/{prefix}_TEST'.format(prefix=prefix), header=None)
    len_aa = len(aa)
    aa_labels = aa.iloc[:,0]
    aa = np.array(aa.iloc[:,1:])
    aa = aa.reshape(len_aa, 1, MAX_SEQUENCE_LENGTH)
    # print(np.array(aa_labels))
    # print(np.argmax(model.predict(aa), axis=1))
    millis = int(round(time.time() * 1000))
    model.predict(aa)
    print(int(round(time.time() * 1000)-millis))
    print(classification_report(np.array(aa_labels), np.argmax(model.predict(aa), axis=1), 
             target_names=[l for l in read_dictionary.values()]),
          file=open('weights/{prefix}_{model_num}_report.txt'.format(prefix=prefix, model_num=model_num), "w"))





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datasetindex', help='Dataset index', type=int, required=True)
    parser.add_argument('-m', '--model_num', help='LSTM, FCN, LSTM-FCN, or ALSTM-FCN', type=int, default=3)
    parser.add_argument('-r', '--run_training', help='Run training or not', type=int, default=0)
    parser.add_argument('-p', '--prefix', help='Prefix to dataset', type=str, default='Output')
    parser.add_argument('-e', '--epochs', help='Epochs for training', type=int, default=500)
    parser.add_argument('-b', '--batch_size', help='Batch size for training', type=int, default=None)
    parser.add_argument('-l', '--lstm_cell_num', help='Number of LSTM cells', type=int, default=None)
    args = parser.parse_args()

    if not args.datasetindex:
        parser.error('No dataset index given, add --datasetindex')

    trainer(args.datasetindex, args.model_num, args.run_training, args.prefix, args.epochs, args.batch_size, args.lstm_cell_num)