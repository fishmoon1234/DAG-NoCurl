# utility files for testing
# @author: tian gao

import numpy as np
import scipy.linalg as slin
import scipy.sparse as sp
import networkx as nx
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import os
from scipy.special import expit as sigmoid
import glob
import re
import pickle
import math

import logging

def setup_logger( mode = 'debug'):
    if mode == 'debug':
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger('fly')

    return logger

def simulate_random_dag(d: int,
                        degree: float,
                        graph_type: str,
                        w_range: tuple = (0.5, 2.0)) -> nx.DiGraph:
    """Simulate random DAG with some expected degree.

    Args:
        d: number of nodes
        degree: expected node degree, in + out
        graph_type: {erdos-renyi, barabasi-albert, full}
        w_range: weight range +/- (low, high)

    Returns:
        G: weighted DAG
    """
    if graph_type == 'erdos-renyi':
        prob = float(degree) / (d - 1)
        B = np.tril((np.random.rand(d, d) < prob).astype(float), k=-1)
    elif graph_type == 'barabasi-albert':
        m = int(round(degree / 2))
        B = np.zeros([d, d])
        bag = [0]
        for ii in range(1, d):
            dest = np.random.choice(bag, size=m)
            for jj in dest:
                B[ii, jj] = 1
            bag.append(ii)
            bag.extend(dest)
    elif graph_type == 'full':  # ignore degree, only for experimental use
        B = np.tril(np.ones([d, d]), k=-1)

    elif graph_type == 'chain': # ignore degree, only for experimental use
        B = np.zeros([d, d])
        B[np.arange(d-1), np.arange(d-1)+1] = 1

    else:
        raise ValueError('unknown graph type')
    # random permutation
    P = np.random.permutation(np.eye(d, d))  # permutes first axis only
    B_perm = P.T.dot(B).dot(P)
    U = np.random.uniform(low=w_range[0], high=w_range[1], size=[d, d])
    U[np.random.rand(d, d) < 0.5] *= -1
    W = (B_perm != 0).astype(float) * U
    G = nx.DiGraph(W)
    return G

def simulate_sem_nonlinear(G: nx.DiGraph,
                 n: int,
                 x_dims: int,
                 sem_type: str,
                 linear_type: str,
                 noise_scale: float = 1.0) -> np.ndarray:
    """Simulate samples from SEM with specified type of noise.

    Args:
        G: weigthed DAG
        n: number of samples
        sem_type: {linear-gauss,linear-exp,linear-gumbel}
        noise_scale: scale parameter of noise distribution in linear SEM

    Returns:
        X: [n,d] sample matrix
    """
    W = nx.to_numpy_array(G)
    d = W.shape[0]
    X = np.zeros([n, d, x_dims])
    ordered_vertices = list(nx.topological_sort(G))
#    topW = np.copy(W)
#    topW[topW < 0.] = 0
#    ordered_vertices = list(nx.topological_sort(nx.DiGraph(topW)))
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = list(G.predecessors(j))
        if linear_type == 'linear':
            eta = X[:, parents, 0].dot(W[parents, j])
        elif linear_type == 'nonlinear_1':
            eta = np.cos(X[:, parents, 0] + 1).dot(W[parents, j])
        elif linear_type == 'nonlinear_2':
            eta = (X[:, parents, 0]+0.5).dot(W[parents, j])
        else:
            raise ValueError('unknown linear data type')

        if sem_type == 'linear-gauss':
            if linear_type == 'linear':
                X[:, j, 0] = eta + np.random.normal(scale=noise_scale, size=n)
            elif linear_type == 'nonlinear_1':
                X[:, j, 0] = eta + np.random.normal(scale=noise_scale, size=n)
            elif linear_type == 'nonlinear_2':
                X[:, j, 0] = 2.*np.sin(eta) + eta + np.random.normal(scale=noise_scale, size=n)
        elif sem_type == 'linear-exp':
            if linear_type == 'linear':
                X[:, j, 0] = eta + np.random.exponential(scale=noise_scale, size=n)
            elif linear_type == 'nonlinear_1':
                X[:, j, 0] = eta + np.random.exponential(scale=noise_scale, size=n)
            elif linear_type == 'nonlinear_2':
                X[:, j, 0] = 2.*np.sin(eta) + eta + np.random.exponential(scale=noise_scale, size=n)
        elif sem_type == 'linear-gumbel':
            X[:, j, 0] = eta + np.random.gumbel(scale=noise_scale, size=n)
        else:
            raise ValueError('unknown sem type')
    if x_dims > 1 :
        for i in range(x_dims-1):
            X[:, :, i+1] = np.random.normal(scale=noise_scale, size=1)*X[:, :, 0] + np.random.normal(scale=noise_scale, size=1) + np.random.normal(scale=noise_scale, size=(n, d))
        X[:, :, 0] = np.random.normal(scale=noise_scale, size=1) * X[:, :, 0] + np.random.normal(scale=noise_scale, size=1) + np.random.normal(scale=noise_scale, size=(n, d))
    return X

def simulate_sem(G: nx.DiGraph,
                 n: int,
                 sem_type: str,
                 noise_scale: float = 1.0) -> np.ndarray:
    """Simulate samples from SEM with specified type of noise.

    Args:
        G: weigthed DAG
        n: number of samples
        sem_type: {linear-gauss,linear-exp,linear-gumbel}
        noise_scale: scale parameter of noise distribution in linear SEM

    Returns:
        X: [n,d] sample matrix
    """
    W = nx.to_numpy_array(G)
    d = W.shape[0]
    X = np.zeros([n, d])
    ordered_vertices = list(nx.topological_sort(G))
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = list(G.predecessors(j))
        #eta = np.cos(X[:, parents]+1.).dot(W[parents, j])  # [n,]
        eta = (X[:, parents]).dot(W[parents, j])
        if sem_type == 'linear-gauss':
            X[:, j] = eta + np.random.normal(scale=noise_scale, size=n)
        elif sem_type == 'linear-exp':
            X[:, j] = eta + np.random.exponential(scale=noise_scale, size=n)
        elif sem_type == 'linear-gumbel':
            X[:, j] = eta + np.random.gumbel(scale=noise_scale, size=n)
        else:
            raise ValueError('unknown sem type')
    return X

def simulate_sem_multid(G: nx.DiGraph,
                 n: int,
                 x_dims: int,
                 sem_type: str,
                 noise_scale: float = 1.0) -> np.ndarray:
    """Simulate samples from SEM with specified type of noise.

    Args:
        G: weigthed DAG
        n: number of samples
        sem_type: {linear-gauss,linear-exp,linear-gumbel}
        noise_scale: scale parameter of noise distribution in linear SEM

    Returns:
        X: [n,d] sample matrix
    """
    W = nx.to_numpy_array(G)
    d = W.shape[0]
    X = np.zeros([n, d, x_dims])
    ordered_vertices = list(nx.topological_sort(G))
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = list(G.predecessors(j))
        # eta = (np.sin(X[:, parents])+1.).dot(W[parents, j])  # [n,]
        eta = X[:, parents, 0].dot(W[parents, j])
        if sem_type == 'linear-gauss':
            X[:, j, 0] = eta + np.random.normal(scale=noise_scale, size=n)
        elif sem_type == 'linear-exp':
            X[:, j, 0] = eta + np.random.exponential(scale=noise_scale, size=n)
        elif sem_type == 'linear-gumbel':
            X[:, j, 0] = eta + np.random.gumbel(scale=noise_scale, size=n)
        else:
            raise ValueError('unknown sem type')
    for i in range(x_dims-1):
        X[:, :, i+1] = np.random.normal(scale=noise_scale, size=1)*X[:, :, 0] + np.random.normal(scale=noise_scale, size=1) + np.random.normal(scale=noise_scale, size=(n, d))
    X[:, :, 0] = np.random.normal(scale=noise_scale, size=1) * X[:, :, 0] + np.random.normal(scale=noise_scale, size=1) + np.random.normal(scale=noise_scale, size=(n, d))
 #       for j in ordered_vertices:
 #           parents = list(G.predecessors(j))
#            eta = X[:, parents, i].dot(W[parents, j])
#            if sem_type == 'linear-gauss':
#                X[:, j, i] = eta + np.random.normal(scale=noise_scale, size=n)
#            elif sem_type == 'linear-exp':
#                X[:, j, i] = eta + np.random.exponential(scale=noise_scale, size=n)
#            elif sem_type == 'linear-gumbel':
#                X[:, j, i] = eta + np.random.gumbel(scale=noise_scale, size=n)
#            else:
#                raise ValueError('unknown sem type')
    return X

def count_accuracy(G_true: nx.DiGraph,
                   G: nx.DiGraph,
                   G_und: nx.DiGraph = None) -> tuple:
    """Compute FDR, TPR, and FPR for B, or optionally for CPDAG B + B_und.

    Args:
        G_true: ground truth graph
        G: predicted graph
        G_und: predicted undirected edges in CPDAG, asymmetric

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    B_true = nx.to_numpy_array(G_true) != 0
    B = nx.to_numpy_array(G) != 0
    B_und = None if G_und is None else nx.to_numpy_array(G_und)
    d = B.shape[0]
    # linear index of nonzeros
    if B_und is not None:
        pred_und = np.flatnonzero(B_und)
    pred = np.flatnonzero(B)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    if B_und is not None:
        # treat undirected edge favorably
        true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
        true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    if B_und is not None:
        false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
        false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    if B_und is not None:
        pred_size += len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    B_lower = np.tril(B + B.T)
    if B_und is not None:
        B_lower += np.tril(B_und + B_und.T)
    pred_lower = np.flatnonzero(B_lower)
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    # print('extra %f + missing %f + reverse %f' % ( len(extra_lower), len(missing_lower), len(reverse)))
    return fdr, tpr, fpr, shd, pred_size #, len(extra_lower), len(missing_lower), len(reverse)

def count_accuracy_new(G_true: nx.DiGraph,
                   G: nx.DiGraph,
                   G_und: nx.DiGraph = None) -> tuple:
    """Compute FDR, TPR, and FPR for B, or optionally for CPDAG B + B_und.

    Args:
        G_true: ground truth graph
        G: predicted graph
        G_und: predicted undirected edges in CPDAG, asymmetric

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    B_true = nx.to_numpy_array(G_true) != 0
    B = nx.to_numpy_array(G) != 0
    B_und = None if G_und is None else nx.to_numpy_array(G_und)
    d = B.shape[0]
    # linear index of nonzeros
    if B_und is not None:
        pred_und = np.flatnonzero(B_und)
    pred = np.flatnonzero(B)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    if B_und is not None:
        # treat undirected edge favorably
        true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
        true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    if B_und is not None:
        false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
        false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    if B_und is not None:
        pred_size += len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    B_lower = np.tril(B + B.T)
    if B_und is not None:
        B_lower += np.tril(B_und + B_und.T)
    pred_lower = np.flatnonzero(B_lower)
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    print('extra %f + missing %f + reverse %f' % ( len(extra_lower), len(missing_lower), len(reverse)))

    return fdr, tpr, fpr, shd, pred_size, len(extra_lower), len(missing_lower), len(reverse)

def get_loss_L2(W, X, loss_type = 'l2'):
    """Evaluate value and gradient of loss."""
    M = X @ W
    # loss_type = self.loss_type
    if loss_type == 'l2':
        R = X - M
        loss = 0.5 / X.shape[0] * (R ** 2).sum()
        G_loss = - 1.0 / X.shape[0] * X.T @ R
    elif loss_type == 'logistic':
        loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
        G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
    elif loss_type == 'poisson':
        S = np.exp(M)
        loss = 1.0 / X.shape[0] * (S - X * M).sum()
        G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
    else:
        raise ValueError('unknown loss type')
    return loss, G_loss

def print_to_file(args,
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
                  search_result = 0,
                  repeat_num = 100
                  ):
    search_string = ['', '_search', '_searchNT', '_searchINP']
    nonzero_string = ['_nonZero', '_zeroM']
    pre_use_l2_string = ['_zeroInit', '_notearInit']

    method_info = str(repeat_num) + args.methods + search_string[search_result] + '_'+ str(args.data_variable_size) + \
                  '_' + args.data_type + '_' + str(args.data_sample_size) + '_' +\
                  str(args.graph_type) + '_' + str(args.graph_sem_type) + \
                  '_' + str(args.graph_degree) + '_hTol_' + str(args.h_tol) + \
                  '_lambda_' + str(args.lambda1) + '_' + str(args.lambda2)

    output_file_name = os.path.join('results', method_info)

    f = open(output_file_name + '.txt', 'w')

    f.write(method_info + '\n')
    # f.write(args)
    print(args, file=f)

    n = math.sqrt(repeat_num)

    print_string = 'Mean Time: ' + str(np.mean(result_time)) + '; std +-' \
                  + str(np.std(result_time).item()/n)

    f.write(print_string + '\n')

    print_string = 'Mean loss: ' + str(np.mean(result_loss)) + '; std +-' \
                  + str(np.std(result_loss).item()/n)
    f.write(print_string + '\n')
    print_string = 'Mean SHD: ' + str(np.mean(result_shd)) + '; std +-' \
                  + str(np.std(result_shd).item()/n)
    f.write(print_string + '\n')
    print_string = 'Mean nnz: ' + str(np.mean(result_nnz)) + '; std +-' \
                  + str(np.std(result_nnz).item()/n)
    print_string = 'Mean tpr: ' + str(np.mean(result_tpr)) + '; std +-' \
                   + str(np.std(result_tpr).item() / n)
    print_string = 'Mean fpr: ' + str(np.mean(result_fpr)) + '; std +-' \
                   + str(np.std(result_fpr).item() / n)
    print_string = 'Mean fdr: ' + str(np.mean(result_fdr)) + '; std +-' \
                   + str(np.std(result_fdr).item() / n)
    f.write(print_string + '\n')
    # print_string = 'Mean Time: ' + str(np.mean(result_fdr)) + '; std +-' \
    #               + str(np.std(result_fdr).item())
    # f.write(print_string + '\n')
    print_string = 'Mean h(A): ' + str(np.mean(result_h)) + '; std +-' \
                  + str(np.std(result_h).item()/n)
    f.write(print_string + '\n')
    print_string = 'Mean extra edge: ' + str(np.mean(result_extra)) + '; std +-' \
                  + str(np.std(result_extra).item()/n)
    f.write(print_string + '\n')
    print_string = 'Mean missing edge: ' + str(np.mean(result_missing)) + '; std +-' \
                  + str(np.std(result_missing).item()/n)
    f.write(print_string + '\n')
    print_string = 'Mean reverse edge: ' + str(np.mean(result_reverse)) + '; std +-' \
                   + str(np.std(result_reverse).item()/n)
    f.write(print_string + '\n')

    f.write( '&'+ str(np.mean(result_shd).item())+ ' $\pm$' + str(np.std(result_shd).item()/n) +
                    ' & '+ str(np.mean(result_nnz).item()) + '$\pm$' + str(np.std(result_nnz).item()/n) +
                    ' & ' + str(np.mean(result_time).item()) + '$\pm$' + str(np.std(result_time).item()/n) )
    f.write('\n')
    f.close()
    
    # Save DataFrame of all results
    df = np.hstack((result_time, result_loss, result_shd, result_nnz,
                    result_tpr, result_fpr, result_fdr,
                    result_h, result_extra, result_missing, result_reverse))
    df = pd.DataFrame(df, columns=['time','lossW', 'SHD', 'nnz',
                                   'tpr', 'fpr', 'fdr',
                                   'h', 'extra', 'missing', 'reverse'])
    df.to_pickle(output_file_name + '.pkl')
