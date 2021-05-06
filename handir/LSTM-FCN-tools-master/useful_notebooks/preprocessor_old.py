import os
import argparse
import math as m
from collections import Counter
import numpy as np
import pandas as pd


def preprocessor(datasetname, verbose, prefix, percent, uppercats):

    """
    Loading and preprocessing dataset

    datasetname = Dataset name (including path)
    verbose = Verbose or not
    prefix = Prefix to dataset
    percent = Percent dataset split (training); default = 0.67
    uppercats = Use an upper-level set of categories or not
    """

    # Read original dataset
    if os.path.exists(datasetname): # -d
        df = pd.read_csv(datasetname, header=None, encoding='latin-1')
    else:
        raise FileNotFoundError('File %s not found!' % (datasetname))

    # Making sure NAs are taken care of
    df.dropna(axis=1, how='all', inplace=True)
    df.fillna(0, inplace=True)

    df = df.rename(columns={df.columns[0]: 'label'})
    df = df.reset_index(drop=True)


    # If there's the need to train/test using the dataset relabeled with an upper-level set of categories
    if uppercats == 1: # -u
        upper_cats = dict.fromkeys(['DQ','DQ1','DQ2','DQ3','MeP','MeP1','MeP2','MeP3','PQ','PQ1','PQ2','PQ3'], 'HandP')
        upper_cats.update(dict.fromkeys(['BPA','BPA1','BPA2','BPA3','NP','NP1','NP2','NP3'], 'Ind'))
        upper_cats.update(dict.fromkeys(['Cd','Cd1','Cd2','Cd3','Cu','Cu1','Cu2','Cu3','HMM','HMM1','HMM2','HMM3',
                                         'Hg','Hg1','Hg2','Hg3','Pd','Pd1','Pd2','Pd3'], 'HM'))
        upper_cats.update(dict.fromkeys(['SW','SW0'], 'SW'))

        df = df.replace(upper_cats)


    # Dealing with labels and classes
    classes = dict(enumerate(np.unique(df.label)))
    classes = {v: k for k, v in classes.items()}
    df.label = df.label.map(classes)
    df_classes_counts = Counter(df.label).values()
    # Save dictionary
    np.save('../data/{}_labels_dict.npy'.format(prefix), classes) 


    # Spliting the dataset
    ave_samp_class = m.floor(df.shape[0] / len(np.unique(df.label)))
    percent_train = percent # -c

    df1 = pd.DataFrame([])
    for i in np.unique(df.label):
        df1 = df1.append(df[df.label == i].sample(m.floor(ave_samp_class * percent_train)))

    df2 = df[~df.index.isin(df1.index)]
    df1_classes_counts = Counter(df1.label).values()
    df2_classes_counts = Counter(df2.label).values()

    # Saving to CSV files
    df1.to_csv('../data/{}_TRAIN'.format(prefix), header=None, index_label=None, index=None) # -p
    df2.to_csv('../data/{}_TEST'.format(prefix), header=None, index_label=None, index=None)

    if verbose == 1: # -v
        print()
        print("Finished loading dataset...")
        print("------------------------")
        print("Number of samples:  ", df.shape[0])
        print("Number of classes:  ", len(np.unique(df.label)))
        print("Number of samples per class:  ", df_classes_counts)
        print("Sequence length:  ", df.shape[-1])
        print("------------------------")
        print("Number of samples in training set:  ", df1.shape[0])
        print("Number of classes in training set:  ", len(df1_classes_counts))
        print("Number of training samples per class:  ", df1_classes_counts)
        print("------------------------")
        print("Number of samples in test set:  ", df2.shape[0])
        print("Number of classes in test set:  ", len(df2_classes_counts))
        print("Number of testing samples per class:  ", df2_classes_counts)


    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datasetname', help='Dataset name (including path)', type=str, required=True)
    parser.add_argument('-v', '--verbose', help='Verbose or not', type=int, default=1)
    parser.add_argument('-p', '--prefix', help='Prefix to dataset', type=str, default='Output')
    parser.add_argument('-c', '--percent', help='Percent dataset split (training)', type=float, default=0.67)
    parser.add_argument('-u', '--uppercats', help='Relabel with upper-level category', type=int, default=0)
    args = parser.parse_args()

    if not args.datasetname:
        parser.error('No dataset name given, add --datasetname')

    preprocessor(args.datasetname, args.verbose, args.prefix, args.percent, args.uppercats)
