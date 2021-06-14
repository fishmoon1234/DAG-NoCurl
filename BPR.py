'''
main class for bayesian network penalized regression

Tian Gao, 3/6/2020
'''


import numpy as np
from scipy.linalg import expm
import utils
import networkx as nx
from utils import setup_logger
from tqdm import tqdm

from scipy import linalg

import pandas as pd
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
import networkx as nx

class BPR:
    def __init__(self, args):
        self.args = args
        self.rho_max = args.rho_A_max # augmented Langagragian coefficent
        self.h_tol = args.h_tol
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.train_epochs = args.train_epochs
        self.threshold_A = args.graph_threshold
        self.loss_type = 'l2'

    def fit(self, X, method = 'nocurl'):

        if method == 'notear':
            # --original version of notear
            return self.fit_all(X)

        elif method == 'nocurl':
            return self.fit_all_L2proj(X)

        elif method =='CAM':
            return self.fit_cam(X)

        elif method =='GES':
            return self.fit_ges(X)

        elif method == 'MMPC':
            return self.fit_mmpc(X)

        elif method =='FGS':
            return self.fit_fgs(X)

        else:
            print('method is not support')

    def fit_fgs(self, X):
        '''FGS version'''

        '''FGS version'''
        from fges_continuous_yyu import fit_FGS

        d = X.shape[1]
        trueG = nx.to_numpy_array(self.ground_truth_G)
        A = fit_FGS(X, trueG, d, self.pc)

        lossA = self._loss_L2(A, X, loss_type = 'l2')

        return A, [-1], [], [lossA[0]]


    def fit_cam(self, X):
        import cdt
        model = cdt.causality.graph.CAM(score='nonlinear', cutoff=0.001,
                                        variablesel=True, selmethod='gamboost',
                                        pruning=True, prunmethod='gam',
                                        njobs=None, verbose=None)  # causal additive model: Guasisan process + additive noise
        # model = cdt.causality.graph.LiNGAM()  # Linear Non-Gaussian Acyclic model + addtive noise
        #

        data_frame = pd.DataFrame(X)
        output_graph_nc = model.predict(data_frame)
        A = nx.adjacency_matrix(output_graph_nc).todense()

        A = np.asarray(A).astype(np.float64)

        lossA = self._loss_L2(A, X, loss_type='l2')

        return A, [-1], [],[lossA[0]]

    def fit_mmpc(self, X):
        import cdt
        model = cdt.causality.graph.bnlearn.MMPC()

        data_frame = pd.DataFrame(X)
        output_graph_nc = model.predict(data_frame)
        A = nx.adjacency_matrix(output_graph_nc).todense()
        A = np.asarray(A).astype(np.float64)

        lossA = self._loss_L2(A, X, loss_type='l2')

        return A, [-1], [], [lossA[0]]

    def fit_ges(self, X):
        import cdt
        model = cdt.causality.graph.GES()
        data_frame = pd.DataFrame(X)
        output_graph_nc = model.predict(data_frame)
        A = nx.adjacency_matrix(output_graph_nc).todense()
        A = np.asarray(A).astype(np.float64)

        lossA = self._loss_L2(A, X, loss_type='l2')

        return A, [-1], [], [lossA[0]]


    def fit_aug_lagr_AL2proj(self, X, hTol, lambda1, lambda2, threshold):

        def relu(x, derivative=False, alpha=0.1):
            rel = x * (x > 0)
            if derivative:
                return (x > 0)*1
            return rel


        def _h(w):
            W = w.reshape([d, d])
            M = np.eye(d) + W * W / d  # (Yu et al. 2019)
            E = np.linalg.matrix_power(M, d - 1)
            h = (E.T * M).sum() - d
            G_h = E.T * W * 2
            return h, G_h

        def _loss(W):

            return 0.5 / n * np.square(np.linalg.norm(X.dot(np.eye(d, d) - W), 'fro'))

        def _loss_comp(W):

            return 0.5 / n * np.square(np.linalg.norm(X.dot(np.eye(d, d) - W), axis=0))

        def _prefunc(w):
            W = w.reshape([d, d])
            loss = 0.5 / n * np.square(np.linalg.norm(X.dot(np.eye(d, d) - W), 'fro'))
            h,_ = _h(W)
            return loss + 0.5 * rho * h * h + alpha * h

        def _pregrad(w):
            W = w.reshape([d, d])
            loss_grad = - 1.0 / n * X.T.dot(X).dot(np.eye(d, d) - W)
            h, Gh = _h(W)
            obj_grad = loss_grad + (rho * h + alpha)*Gh
            return obj_grad.flatten()

        def build_W(w, d,totalcount):
        # build w
            w1 = np.zeros([d, d])
            lower_index = np.tril_indices(d, -1)
            w1[lower_index] = w[:totalcount]

            return w1 + w1.T - np.diag(w1.diagonal())

        def build_phi(w, totalcount):
            phi = w[totalcount:].reshape(-1,1)
            return phi

        def _func(w, choice = 'relu'):
            totalcount = int(d*(d-1)/2)
            loss = 0

        # build parameters
            w1 = build_W(w, d, totalcount)
            phi = build_phi(w, totalcount)

        # compute loss
        # loss2 = 0
            ones = np.ones_like(phi)
            W = np.multiply(w1,  relu(np.matmul(ones, phi.T) - np.matmul(phi, ones.T), False))
            loss += _loss(W)

            return loss

        def _grad(w, choice='relu'):
            totalcount = int(d * (d - 1) / 2)
            w1 = build_W(w, d, totalcount) # np.zeros([d,d])
            phi = build_phi(w, totalcount)
            ones = np.ones_like(phi)
            W_all = np.multiply(w1,  relu(np.matmul(ones, phi.T) - np.matmul(phi, ones.T), False))

            loss_grad = np.zeros(totalcount+d)
            W_grad = - 1.0 / n * X.T.dot(X).dot(np.eye(d, d) - W_all)


        # loss to W
            dF_dW_2= np.multiply(W_grad, relu(np.matmul(ones, phi.T) - np.matmul(phi, ones.T), False))
            dF_dW = dF_dW_2 + dF_dW_2.T


            loss_grad[:totalcount] = dF_dW[np.tril_indices(d, -1)]

        # loss to phi
            dF_dPhi_2= np.multiply( np.multiply(W_grad, w1),  relu(np.matmul(ones, phi.T) - np.matmul(phi, ones.T), True))


            loss_grad[totalcount:] = np.sum(dF_dPhi_2 - dF_dPhi_2.T, axis=0)

            obj_grad = loss_grad #+ (rho * (np.trace(E) - d) + alpha) * E.T * W * 2
            return obj_grad#.flatten()

        def _func1(w):
            loss = _func(w, choice='relu')
            return loss

        def _grad1(w):
            obj_grad = _grad(w, choice='relu')
            return obj_grad

        def _func_w(w_input, *args):
        #         W = w[0:d*d].reshape([d, d])

            choice = args[0] # 'relu'
            totalcount = int(d * (d - 1) / 2)
            loss = 0

        # build parameters
            phi = args[1]
            w = np.zeros(totalcount + d)
            w[:totalcount] = w_input
            w[totalcount:totalcount+d-1] = phi
            w1 = build_W(w, d, totalcount)
            phi = build_phi(w, totalcount)

        # compute loss
        # loss2 = 0
            ones = np.ones_like(phi)
            W = np.multiply(w1, relu(np.matmul(ones, phi.T) - np.matmul(phi, ones.T), False))

            return  _loss(W)

        def build_w_inv(A, phi, d,totalcount):
        # build w
            w1 = np.zeros([d, d])
            for i in range(d):
                for j in range(d):
                    if ((relu(phi[j]-phi[i])>1e-8)):
                        w1[i,j] = A[i,j]/relu(phi[j]-phi[i])
                    else:
                        w1[i,j] = 0
            w = (w1 + w1.T)/2.
            wnew = np.zeros(totalcount)
            lower_index = np.tril_indices(d, -1)
            wnew[:totalcount] = w[lower_index]

            return wnew

        def _grad_w(w_input, *args):
            #        W = w[0:d*d].reshape([d, d])
            choice = args[0]
            phi = args[1]

            totalcount = int(d * (d - 1) / 2)

            w = np.zeros(totalcount + d)
            w[:totalcount] = w_input
            w[totalcount:totalcount+d-1] = phi

            w1 = build_W(w, d, totalcount)  # np.zeros([d,d])
            phi = build_phi(w, totalcount)
            ones = np.ones_like(phi)
            W_all = np.multiply(w1, relu(np.matmul(ones, phi.T) - np.matmul(phi, ones.T), False))

            loss_grad = np.zeros(totalcount)
            W_grad = - 1.0 / n * X.T.dot(X).dot(np.eye(d, d) - W_all)

        # loss to W
            dF_dW_2 = np.multiply(W_grad, relu(np.matmul(ones, phi.T) - np.matmul(phi, ones.T), False))
            dF_dW = dF_dW_2 + dF_dW_2.T
            loss_grad[:totalcount] = dF_dW[np.tril_indices(d, -1)]
            obj_grad = loss_grad
            return obj_grad  # .flatten()

        n, d = X.shape
        totalcount = int(d * (d - 1) / 2)
        w_est, w_best = np.zeros(totalcount + d), np.zeros(totalcount + d)
        bnds = [(None, None) for i in range(totalcount + d)]
        bnds[totalcount + d - 1] = (0, 0)
        bndsw = [(None, None) for i in range(totalcount)] #bounds for w in g(phi)
        rho = 0.0
        alpha = lambda1
        prew = np.zeros(d * d)
        prebnds = [(0, 0) if i == j else (None, None) for i in range(d) for j in range(d)]

        presol = sopt.minimize(_prefunc, prew, method='L-BFGS-B', jac=_pregrad, bounds=prebnds, options={'ftol': hTol})
        prew = np.copy(presol.x)

        rho = 0.0
        alpha = lambda2

        presol = sopt.minimize(_prefunc, prew, method='L-BFGS-B', jac=_pregrad, bounds=prebnds, options={'ftol': hTol})
        prew = np.copy(presol.x)

        prew[np.abs(prew) < 0.3] = 0

        preX = np.sign(np.abs(prew.reshape([d, d])))
        preX = np.sign(linalg.expm(preX)-np.identity(d))
        asymX = preX - preX.transpose()

        preM = np.ones([d-1,d-1])
        for i in range(d-1):
            preM[i,i] = -(d-1)
        preb = np.sum(asymX, axis=1)

        prephi = np.linalg.solve(preM, preb[0:d-1])


        w_est = np.zeros(totalcount + d)
        w_est[totalcount:totalcount + d - 1] = prephi
        w_phi = np.zeros(totalcount)


###############################
#standard 2-step version
        solw1 = sopt.minimize(_func_w, w_phi, args=('relu', prephi), method='L-BFGS-B', jac=_grad_w, bounds=bndsw, options={'ftol': hTol})
        w_est[0:totalcount] = solw1.x

#full version
#        w_est[0:totalcount + d - 1] = np.random.rand(totalcount + d - 1)
#        sol2 = sopt.minimize(_func1, w_est, method='L-BFGS-B', jac=_grad1, bounds=bnds, options={'ftol': hTol})
#        w_est = np.copy(sol2.x)
################################

        w1 = build_W(w_est, d, totalcount)
        phi = build_phi(w_est, totalcount)

        ones = np.ones_like(phi)
        W = np.multiply(w1, relu(np.matmul(ones, phi.T) - np.matmul(phi, ones.T), False))

        lossW = _loss(W)

#    W[np.abs(W) < threshold] = 0

        return W, lossW


    def fit_all_L2proj(self, X):
        Wstar, lossW = self.fit_aug_lagr_AL2proj(X,
                                    hTol=self.h_tol,
                                    lambda1=self.lambda1,
                                    lambda2=self.lambda2,
                                    threshold=self.threshold_A)

        # compare how sparse it is
        W_original = Wstar.copy()
        W_original[np.abs(W_original) < 1e-5] = 0
        num_non_zero  = np.sum( np.abs(W_original)>0)

        Wstar[np.abs(Wstar) < self.threshold_A] = 0



        return Wstar, [0.], [num_non_zero],  [lossW]


    def fit_all(self,X):
        '''original notear'''

        def _h(w):
            W = w.reshape([d, d])
            M = np.eye(d) + W * W / d  # (Yu et al. 2019)
            E = np.linalg.matrix_power(M, d - 1)
            h = (E.T * M).sum() - d
            return h # np.trace(slin.expm(W * W)) - d

        def _func(w):
            W = w.reshape([d, d])
            loss = 0.5 / n * np.square(np.linalg.norm(X.dot(np.eye(d, d) - W), 'fro'))
            h = _h(W)
            return loss + 0.5 * rho * h * h + alpha * h

        def _loss(w):
            W = w.reshape([d, d])
            loss = 0.5 / n * np.square(np.linalg.norm(X.dot(np.eye(d, d) - W), 'fro'))
            return loss

        def _grad(w):
            W = w.reshape([d, d])
            loss_grad = - 1.0 / n * X.T.dot(X).dot(np.eye(d, d) - W)
            M = np.eye(d) + W * W / d
            E = np.linalg.matrix_power(M, d - 1) 
            obj_grad = loss_grad + (rho * _h(w) + alpha) * E.T * W * 2
            return obj_grad.flatten()

        T = int(self.train_epochs)
        n, d = X.shape
        w_est, w_new = np.zeros(d * d), np.zeros(d * d)
        rho, alpha, h, h_new = 1.0, 0.0, np.inf, np.inf
        alpha_iter = np.zeros(T + 1)
        rho_iter = np.zeros(T + 1)
        bnds = [(0, 0) if i == j else (None, None) for i in range(d) for j in range(d)]
        for t in range(T):
            while rho < self.rho_max:
                sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=_grad, bounds=bnds)
                w_new = sol.x
                h_new = _h(w_new)
                if h_new > 0.25 * h:
                    rho *= 10
                else:
                    break
            w_est, h = w_new, h_new
            alpha += rho * h
            alpha_iter[t + 1] = alpha
            rho_iter[t + 1] = rho

            if h <= self.h_tol:
                break

        lossA = _loss(w_est)

        w_est[np.abs(w_est) < self.threshold_A] = 0
        A = w_est.reshape([d, d])
        # return
#        lossA = self._loss_L2(w_est, X, loss_type='l2')

        return A, [h], alpha_iter, [lossA]


def sigmoid(x, derivative=False):
    D=10.
    sigm = 1. / (1. + np.exp(-D*(x)))
    if derivative:
        return D*sigm * (1. - sigm)
    return sigm

def softrelu(x, derivative=False, alpha=0.1):
    D=10.
    rel = np.log((1. + np.exp((D*(x+0.)))))/D #x * (x > 0)
    if derivative:
        return 1./ (1. + np.exp(-(D*(x+0.))))#(x > 0)*1
    return rel

def elu(x, derivative=False, alpha=0.1):
    rel = x * (x > 0) + alpha*(np.exp(x)-1) * (x<0)
    if derivative:
        return (x > 0)*1 + alpha*np.exp(x) * (x<0)
    return rel


def relu(x, derivative=False, alpha=0.1):
    rel = x * (x > 0)
    if derivative:
        return (x > 0)*1
    return rel


def activation (x, derivative = False, choice='relu'):
    if choice == 'sigmoid':
        return sigmoid(x, derivative)
    elif choice == 'softrelu':
        return softrelu(x, derivative)
    elif choice == 'relu':
        return relu(x, derivative)
    elif choice == 'relubar':
        return relu(x, derivative)
    elif choice == 'elu':
        return elu(x, derivative)
