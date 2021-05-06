from sklearn.metrics.pairwise import pairwise_distances
from collections import Counter, namedtuple
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix
from functools import partial
from fastdtw import fastdtw
from tqdm import tqdm
import numpy as np
import argparse
import time
import math
import os
from collections import Counter
import numpy as np
import pandas as pd

SEED = 0
np.random.seed = SEED
import warnings as w
w.simplefilter(action = 'ignore', category = FutureWarning)



def loaddataset(name, train_suffix='_TRAIN', test_suffix='_TEST'):
    data = np.loadtxt('./data/{0}{1}'.format(name, train_suffix), delimiter=',')
    test_data = np.loadtxt('./data/{0}{1}'.format(name, test_suffix), delimiter=',')

    # get labels
    min_label = min(data[:,0])
    labels = np.array(data[:,0], dtype=int) - min_label
    min_label = min(test_data[:,0])
    test_labels = np.array(test_data[:,0], dtype=int) - min_label

    # remove labels
    data = data[:,1:]
    test_data = test_data[:,1:]

    class_dist = Counter(labels)

    print('Dataset {} - class distribution of: {}, test:{}'.format(name, class_dist, len(test_labels)))
    K = len(class_dist)
    T = data.shape[1]
    N = len(data)

    r = namedtuple('dataset', 'data, labels, class_dist, test_data, test_labels, T, N, K')
    return r(data, labels, class_dist, test_data, test_labels, T, N, K)


def savedataset(dataset_name, data, labels=None, suffix=''):
    if labels is not None:
        data = np.concatenate([labels[:,np.newaxis], data], axis=1)
    fname = './data/{0}{1}'.format(dataset_name, suffix)
    f = open(fname, 'ab')
    np.savetxt(f, data, fmt='%g', delimiter=',')
    f.close()


def ssg(X, n_epochs=None, eta=None, init_sequence=None, return_f=False):
    # Inputs
    # X is a 3-dim matrix consisting of possibly multivariate time series.
    #   dim 1 runs over the sample time series
    #   dim 2 runs over the length of a time series
    #   dim 3 runs over the dimension of the datapoints of a time series
    #
    # Optional Inputs
    # n_epochs        is the number of epochs
    # eta             is a vector of step sizes, eta(i) is used in the i-th update
    # init_sequence   if None  --> use a random sample of X
    #                 if > 0   --> use X[init_sequence]
    #                 if <= 0  --> use medoid of X
    #                 if it is a time series --> use it
    # return_f        if True  --> Frechet variations for each epoch are returned
    #
    # Outputs
    # z               the solution found by SSG (an approximate sample mean under dynamic time warping)
    # f               Vector of Frechet variations. Is only returned if return_f=True
    
    N = X.shape[0]  # number of samples
    d = X.shape[2]  # dimension of data

    if n_epochs is None:
        n_updates = 1000
        n_epochs = int(np.ceil(n_updates / N))

    if eta is None:
        eta = np.linspace(0.1, 0.005, N)

    # initialize mean z
    if init_sequence is None:
        z = X[np.random.randint(N)]

    elif init_sequence > 0:
        z = X[int(init_sequence)]

    elif init_sequence <= 0:
        z = medoid_sequence(X)

    if return_f:
        f = np.zeros(n_epochs + 1)
        f[0] = frechet(z, X)

    # stochastic subgradient optimization
    with tqdm(total=n_epochs * N) as pbar:
        for k in range(1, n_epochs + 1):
            perm = np.random.permutation(N)
            for i in range(1, N + 1):
                pbar.update(1)
                x_i = X[perm[i - 1]]
                _, p = dtw(z, x_i, path=True)

                W, V = get_warp_val_mat(p)
                
                subgradient = 2 * (V * z - W.dot(x_i))

                c = (k - 1) * N + i
                if c <= eta.shape[0]:
                    lr = eta[c - 1]
                else:
                    lr = eta[-1]

                # update rule
                z = z - lr * subgradient

            if return_f:
                f[k] = frechet(z, X)

    if return_f:
        f = f[0:n_epochs + 1]
        return z, f

    else:
        return z


def dtw(x, y, path=False):
    # Local Variables: C, d, C_diag, k, C_d, m, N, p, C_r, y, x, n, D
    # Function calls: pdist2, min, cumsum, M, nargout, sqrt, zeros, dtw, size
    # %DTW dynamic time warping for multidimensional time series
    # %
    # % Input
    # % x:  [n x d]               d dimensional time series of length n
    # % y:  [m x d]               d dimensional time series of length m
    # %
    # % Output
    # % d:  [1 x 1]               dtw(x,y) with local Euclidean distance
    # % p:  [L x 2]   (optional)  warping path of length L
    # %
    # %
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    N, d = x.shape
    M, _ = y.shape
    D = cdist(x, y) ** 2
    
    C = np.zeros((N, M))
    C[:, 0] = np.cumsum(D[:, 0])
    C[0, :] = np.cumsum(D[0, :])
    
    for n in range(1, N):
        for m in range(1, M):
            C[n, m] = D[n, m] + min(C[n - 1, m - 1], C[n - 1, m], C[n, m - 1])

    d = np.sqrt(C[N - 1, M - 1])
    
    # % compute warping path p
    if path:
        n = N - 1
        m = M - 1
        p = np.zeros((N + M - 1, 2))
        p[-1, :] = (n, m)
        k = 1

        while n + m > 0:
            if n == 0:
                m = m - 1
            elif m == 0:
                n = n - 1

            else:
                C_diag = C[n - 1, m - 1]
                C_r = C[n, m - 1]
                C_d = C[n - 1, m]
                if C_diag <= C_r:
                    if C_diag <= C_d:
                        n = n - 1
                        m = m - 1
                    else:
                        n = n - 1

                elif C_r <= C_d:
                    m = m - 1

    else:
        n = n - 1
        p[-1 - k, :] = (n, m)
        k = k + 1
        p = p[-1 - k + 1:, :]
        
    return d, p


def frechet(x, X):
    # Local Variables: dist, f, i, N, X, x
    # Function calls: Frechet, length, dtw
    N = X.shape[0]
    f = 0
    for i in range(N):
        dist = dtw(x, X[i])
        f = f + dist ** 2

    f = f / N
    return f


def medoid_sequence(X):
    # Local Variables: f, i, f_min, N, i_min, X, x
    # Function calls: Frechet, length, medoidSequence, inf
    # MEDOIDSEQUENCE returns medoid of X
    #  A medoid is an element of X that minimizes the Frechet function
    #  among all elements in X
    N = X.shape[0]
    f_min = np.inf
    i_min = 0
    for i in range(N):
        f = frechet(X[i], X)
        if f < f_min:
            f_min = f
            i_min = i

    x = X[i_min]
    return x


def get_warp_val_mat(p):
    # Local Variables: m, L, n, p, W, V
    # Function calls: length, ones, sparse, getWarpingAndValenceMatrix, sum
    #  W is the (sparse) warping matrix of p
    #  V is a vector representing the diagonal of the valence matrix
    L = p.shape[0]
    N = int(p[-1, 0]) + 1
    M = int(p[-1, 1]) + 1
    W = coo_matrix((np.ones(L), (p[:, 0], p[:, 1])), shape=(N, M)).toarray()
    V = np.sum(W, axis=1, keepdims=True)
    return W, V

def fastdtw_(x, y):
    return fastdtw(x.copy(), y.copy())[0]


def create_new_data(data, count, k=1, ssg_epochs=None):
    cluster_ind = np.random.choice(len(data), size=count, replace=False)
    clusters = data[cluster_ind]
    alloc = np.zeros(len(data))

    # K-Means clustering
    for _ in range(k):
        dists = pairwise_distances(data, clusters, metric=fastdtw_, n_jobs=4) # 4
        alloc = dists.argmin(1)

        new_clusters = []
        for j, cluster in enumerate(clusters):
            if list(alloc).count(j) < 2:
                alloc[alloc == j] = -1
                continue
            d = data[alloc == j]
            z = ssg(d[:,np.newaxis], return_f=False, n_epochs=ssg_epochs)
            new_clusters.append(z[0])
        clusters = np.array(new_clusters)
    return clusters, alloc


def expand_data_set(data, labels, n_reps, n_base, k=1, ssg_epochs=None, callback=None):
    old_data_offset = len(data)
    for i in range(n_reps):
        for label in Counter(labels).keys():
            l_data = data[labels == label]
            count = math.ceil(len(l_data) / n_base)
            new_clusters, _ = create_new_data(l_data, count + 1, k, ssg_epochs)
            if new_clusters.size:
                new_labels = np.ones(len(new_clusters)) * label
                callback(data=new_clusters, labels=new_labels)
                print('{} new data points for label {}'.format(len(new_clusters), label))
                data = np.concatenate([data, new_clusters])
                labels = np.concatenate([labels, new_labels])
    return data[old_data_offset:], labels[old_data_offset:]


def spawn(datasetname, n_reps, n_base=4, k=1, ssg_epochs=None, input_suffix='_TRAIN', output_suffix='_EXP_TRAIN'):
    if os.path.exists('./data/{0}{1}'.format(datasetname, output_suffix)):
        print('WARNING FILE EXISTS WITH SUFFIX:', output_suffix)
    data, labels, class_dist, _, _, _, N, K = loaddataset(datasetname, input_suffix)
    print('expanding {} from {} datapoints, with class distribution of: {}'.format(datasetname, 
                                                                                   len(data),
                                                                                   class_dist))
    print('upper bound for data-points generated:', (N / n_base) * n_reps)
    start = time.time()
    save = partial(savedataset, suffix=output_suffix, dataset_name=datasetname)
    expanded_data_set, expanded_labels = expand_data_set(data, labels, n_reps, n_base, k, ssg_epochs, save)
    savedataset(datasetname, expanded_data_set, expanded_labels, output_suffix + '.bak')

    print("------------------------")
    print("Summary before:")
    data_before = np.loadtxt('./data/{0}{1}'.format(datasetname, input_suffix), delimiter=',')
    print("Shape: ", data_before.shape)
    print("Labels: ", np.unique(data_before[:,0]))
    print("Count: ", Counter(data_before[:,0]).values())
    print("------------------------")
    print("Summary after:")
    data_after = np.loadtxt('./data/{0}{1}'.format(datasetname, output_suffix), delimiter=',')
    print("Shape: ", data_after.shape)
    print("Labels: ", np.unique(data_after[:,0]))
    print("Count: ", Counter(data_after[:,0]).values())




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datasetname', help='Datasetname', type=str)
    parser.add_argument('-r', '--n_reps', help='# iterations', type=int, default=1)
    parser.add_argument('-b', '--n_base', help='# data-points to average to creating new data point', type=int, default=2)
    parser.add_argument('-k', '--k', help='# iterations for K-means clustering', type=int, default=1)
    parser.add_argument('-s', '--ssg_epochs', help='# iterations for mean calculation with SSG', type=int, default=None)
    parser.add_argument('-i', '--input_suffix', help='Suffix for file to be extended', type=str, default='_TRAIN')
    parser.add_argument('-o', '--output_suffix', help='Suffix for created files', type=str, default='_EXP_TRAIN')

    args = parser.parse_args()

    if not (args.datasetname):
        parser.error('No dataset name given, add --datasetname')

    spawn(args.datasetname, args.n_reps, args.n_base, args.k, args.ssg_epochs, args.input_suffix, args.output_suffix)
