
''''
Main function for traininng DAG NoCurl

'''


from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os
from tqdm import tqdm
import os.path

import math
import scipy.linalg as slin
import numpy as np
import networkx as nx

import utils
import BPR



def get_args():
    parser = argparse.ArgumentParser()

    # -----------data parameters ------
    # configurations
    parser.add_argument('--data_type', type=str, default= 'synthetic',
                        choices=['synthetic', 'nonlinear1', 'nonlinear2', 'nonlinear3'],
                        help='choosing which experiment to do.')
    parser.add_argument('--data_sample_size', type=int, default=1000,
                        help='the number of samples of data')
    parser.add_argument('--data_variable_size', type=int, default=10,
                        help='the number of variables in synthetic generated data')
    parser.add_argument('--graph_type', type=str, default='erdos-renyi',
                        choices=['barabasi-albert','erdos-renyi'],
                        help='the type of DAG graph by generation method')
    parser.add_argument('--graph_degree', type=int, default=3,
                        help='the number of degree in generated DAG graph')
    parser.add_argument('--graph_sem_type', type=str, default='linear-gauss',
                        choices=['linear-gauss','linear-gumbel'],
                        help='the structure equation model (SEM) parameter type')
    parser.add_argument('--x_dims', type=int, default=1, # data dimension
                        help='The number of input dimensions: default 1.')

    # -----------training hyperparameters
    parser.add_argument('--repeat', type=int, default= 100,
                        help='the number of times to run experiments to get mean/std')

    parser.add_argument('--methods', type=str, default='nocurl',
                        choices=['notear',                   # notear
                                 'nocurl',             # dag no curl
                                 'CAM', 'GES', 'MMPC', 'FGS'                           # baselines
                                 ] ,
                        help='which method to test') # BPR_all = notear

    parser.add_argument('--graph_threshold', type=  float, default = 0.3,  # 0.3 is good, 0.2 is error prune
                        help = 'threshold for learned adjacency matrix binarization')
    parser.add_argument('--alpha_A',  type = float, default= 1000., #corresponding to alpha
                        help='coefficient for DAG constraint h(A).')
    parser.add_argument('--rho_A_max', type=float, default=1e+16,  # corresponding to rho, needs to  be >> lambda
                        help='coefficient for notears.')
    parser.add_argument('--h_tol', type=float, default = 1e-8,
                        help='the tolerance of error of h(A) to zero')
    parser.add_argument('--train_epochs', type=int, default= 1e4,
                        help='Max Number of iteration in notears.')
    parser.add_argument('--generate_data', type=int, default=1,
                        help='generate new data or use old data')
    parser.add_argument('--file_name', type = str, default = 'test_')
    parser.add_argument('--save-folder', type=str, default='logs',
                        help='Where to save the trained model, leave empty to not save anything.')
    parser.add_argument('--load-folder', type=str, default='',
                        help='Where to load the trained model if finetunning. ' +
                             'Leave empty to train from scratch')

    # -----------parsing
    args = parser.parse_args()


    return args

def main(args):

    # Generate and import data
    n, d = args.data_sample_size, args.data_variable_size # samples, variable size
    graph_type, degree, sem_type = args.graph_type, args.graph_degree, args.graph_sem_type

    # book keeping for results
    num_trials = args.repeat

    if args.data_type.startswith('nonlinear'):
        num_trials = 5

    result_time = np.zeros((num_trials, 1))
    result_tpr = np.zeros((num_trials, 1))
    result_fpr = np.zeros((num_trials, 1))
    result_shd = np.zeros((num_trials, 1))
    result_nnz = np.zeros((num_trials, 1))
    result_fdr = np.zeros((num_trials, 1))
    result_h = np.zeros((num_trials, 1))
    result_extra = np.zeros((num_trials, 1))
    result_missing = np.zeros((num_trials, 1))
    result_reverse = np.zeros((num_trials, 1))
    result_loss = np.zeros((num_trials, 1))

    result_originalW_nnz = np.zeros((num_trials, 1))
    repeat_write = [10, 20, 50, 100]
    repeat_counter = 0

    for trial_index in tqdm(range(num_trials)):
        file_name = './data/lineardata/' + str(args.data_sample_size) + '_' + str(args.data_variable_size) + '_' \
                    + str(args.graph_type) + '_' + str(args.graph_degree) + '_' \
                    + str(args.graph_sem_type) + '_' + str(trial_index) + '.pkl'
        # load nonlinear data
        if args.data_type.startswith('nonlinear'):
            if int(args.data_type[-1]) > 2 :  # nonlinear 3 and others
                dir = './data/nonlineardata/SFd' + str(args.data_variable_size)
                index = (int(args.data_type[-1])-1)  * 5  + int(trial_index)
                data_file_name = 'data' + str(index) + '.npy'
                true_G_file_name = 'DAG' + str(index) + '.npy'
                with open(os.path.join(dir, data_file_name), 'rb') as handle:
                    X = np.load(handle)
                with open(os.path.join(dir, true_G_file_name), 'rb') as handle:
                    graph = np.load(handle)

            else:
                dir = './data/nonlineardata/d' + str(args.data_variable_size)
                index = int(args.data_type[-1]) * (trial_index+1)
                data_file_name = 'data'+str(index)+'.npy'
                true_G_file_name = 'DAG'+str(index)+'.npy'
                with open(os.path.join(dir, data_file_name), 'rb') as handle:
                    X = np.load(handle)

                with open(os.path.join(dir, true_G_file_name), 'rb') as handle:
                    graph = np.load(handle)

            G = nx.DiGraph(graph)
        elif args.generate_data and not os.path.exists(file_name):
            G = utils.simulate_random_dag(d, degree, graph_type)
            G = nx.DiGraph(G)
            X = utils.simulate_sem(G, args.data_sample_size, sem_type)

            with open(file_name, "wb") as f:
                pickle.dump( (G, X), f)

        else:
            with open(file_name, "rb") as f:
                G, X = pickle.load(f)

        # FOR TO BE 2D, so no nonlinear for now
        if X.ndim > 2: # args.graph_linear_type !='linear':
            X = X[:, :, 0]

        methods = args.methods

        # for method in methods:
        method = methods
        t =  time.time()
        bpr = BPR.BPR(args)

        A, h, alpha, rho = bpr.fit(X, method)

        # check nonzero
        # result_originalW_nnz[trial_index] = alpha[0]

        result_time[trial_index] =  time.time() - t

        result_h[trial_index] = h[-1]

        logger.info('Testing Method Done: %s' % method)

        loss_A_ground_truth = utils.get_loss_L2(nx.to_numpy_array(G), X, 'l2')

        G_est = nx.DiGraph(A)
        logger.info('Solving equality constrained problem ... Done')
        # evaluate
        fdr, tpr, fpr, shd, nnz, extra, missing, reverse = utils.count_accuracy_new(G, G_est)
        logger.info('Accuracy: fdr %f, tpr %f, fpr %f, shd %d, nnz %d',
                 fdr, tpr, fpr, shd, nnz)
        result_shd[trial_index] = shd
        result_nnz[trial_index] = nnz
        result_tpr[trial_index] = tpr
        result_fpr[trial_index] = fpr
        result_fdr[trial_index] = fdr
        result_loss[trial_index] = rho[-1] - loss_A_ground_truth[0]  # offset by ground truth

        result_extra[trial_index] = extra
        result_missing[trial_index] = missing
        result_reverse[trial_index] = reverse

        if trial_index == repeat_write[repeat_counter]:
            utils.print_to_file(args,
                      result_time[:trial_index],
                      result_shd[:trial_index],
                      result_nnz[:trial_index],
                      result_tpr[:trial_index],
                      result_fpr[:trial_index],
                      result_fdr[:trial_index],
                      result_h[:trial_index],
                      result_loss[:trial_index],
                      result_extra[:trial_index],
                      result_missing[:trial_index],
                      result_reverse[:trial_index],
                      search_result=0,
                      repeat_num = repeat_write[repeat_counter]
                      )

            repeat_counter += 1


    logger.info('Accuracy: fdr ' + str(np.mean(result_fdr).item()) + '$\pm$' + str(np.std(result_fdr).item()) +
                ', tpr ' + str(np.mean(result_tpr).item()) + '$\pm$' + str(np.std(result_tpr).item()) +
                ', fpr ' + str(np.mean(result_fpr).item()) + '$\pm$' + str(np.std(result_fpr).item()) +
                ', h ' + str(np.mean(result_h).item()) + '$\pm$' + str(np.std(result_h).item()) +
                ', lossW ' + str(np.mean(result_loss).item()) + '$\pm$' + str(np.std(result_loss).item()) +
                ', shd ' + str(np.mean(result_shd).item()) + '$\pm$' + str(np.std(result_shd).item()) +
                ', nnz ' + str(np.mean(result_nnz).item()) + '$\pm$' + str(np.std(result_nnz).item()) +
                ', time ' + str(np.mean(result_time).item()) + '$\pm$' + str(np.std(result_time).item()))

    logger.info('Edges: extra ' + str(np.mean(result_extra).item()) + '$\pm$' + str(np.std(result_extra).item()) +
                ', missing ' + str(np.mean(result_missing).item()) + '$\pm$' + str(np.std(result_missing).item()) +
                ', reverse ' + str(np.mean(result_reverse).item()) + '$\pm$' + str(np.std(result_reverse).item()) +
                ', original nnz' + str(np.mean(result_originalW_nnz).item()) + '$\pm$' + str(np.std(result_originalW_nnz).item())
                )

    utils.print_to_file(args,
                        result_time,
                        result_shd,
                        result_nnz,
                        result_tpr,
                        result_fpr,
                        result_fdr,
                        result_h,
                        result_loss,
                        result_extra,
                        result_missing,
                        result_reverse,
                        search_result=0,
                        repeat_num = args.repeat
                        )



if __name__ == "__main__":

    args = get_args()
    logger = utils.setup_logger(mode='debug')
    logger.info(args)

    main(args)
    print(args)
