import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(); 
sns.set_style("white"); sns.set_style("ticks"); sns.set_context("paper")
from pylab import rcParams; rcParams['figure.figsize'] = 5, 5

from utils.keras_utils import evaluate_model, set_trainable, visualize_context_vector, visualize_cam
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
import h5py



def evaluator(datasetindex, prefix, model_num, sample_name):

    """
    Evaluating the model on a sample

    datasetindex = Datasetindex
    prefix = Prefix to dataset
    model_num = Model number
    sample_name = The name of the sample file (followed by .csv)
    """

    prefix = prefix # Seawater
    datasetindex = datasetindex # 4
    DATASET_INDEX = datasetindex
    MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH_LIST[DATASET_INDEX]
    NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX]
    TRAINABLE = True


    def generate_model_0():
        ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))
        x = LSTM(128)(ip) # 32 instead of 8
        # x = Dropout(0.8)(x)
        out = Dense(NB_CLASS, activation='softmax')(x)
        model = Model(ip, out)

        return model  # python trainer.py -d 5 -m 0 -r 1 -p 'Seawater2' -e 100 -b 8


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


    def generate_model_2():
        ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))
        x = LSTM(32)(ip)   # 8
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


    def generate_model_3():
        ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))
        x = AttentionLSTM(8)(ip) # 8
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


    optm = Adam(lr=1e-3)
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights('weights/{prefix}_{model_num}_weights.h5'.format(prefix=prefix, model_num=model_num))

    read_dictionary = np.load('./data/{prefix}_labels_dict.npy'.format(prefix=prefix)).item()
    read_dictionary = {v: k for k, v in read_dictionary.items()}


    ### Sample name (the file name before the .csv)
    sample_name = sample_name 

    aa = pd.read_csv('./data/{sample_name}.csv'.format(sample_name=sample_name), header=None)
    aa = aa.iloc[0]
    aa = np.delete(np.array(aa), 0)
    aa = aa.reshape(1, 1, MAX_SEQUENCE_LENGTH)
    aa_df = pd.DataFrame(np.transpose(model.predict(aa)))
    aa_df['Label'] = [read_dictionary[int(x)] for x in aa_df.index]
    aa_df.columns.values[0] = 'Probability' 
    aa_df.set_index('Label', inplace=True)

    print(aa_df.sort_values(by=['Probability'], ascending=False).round(3))

    ax = aa_df.sort_values('Probability').plot(kind='barh', colormap='RdBu', fontsize=16)
    ax.set_xlabel('Probability', fontsize=18)
    ax.set_ylabel('Label', fontsize=18)
    ax.legend().set_visible(False)

    plt.savefig('figures/{sample_name}.png'.format(sample_name=sample_name), bbox_inches='tight')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datasetindex', help='Datasetindex', type=int, default=4)
    parser.add_argument('-p', '--prefix', help='Prefix to dataset', type=str, default='ArrowHead')
    parser.add_argument('-m', '--model_num', help='Relabel with upper-level category', type=int, default=3)
    parser.add_argument('-s', '--sample_name', help='Sample name (prefix of the file)', type=str, default='sample')
    args = parser.parse_args()

    evaluator(args.datasetindex, args.prefix, args.model_num, args.sample_name)