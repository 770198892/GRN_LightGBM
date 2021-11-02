import math
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
import time
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


def get_importances(TS_data, time_points, alpha="from_data", SS_data=None, gene_names=None, regulators='all', param={}):
    time_start = time.time()

    ngenes = TS_data[0].shape[1]

    if alpha is "from_data":
        alphas = estimate_degradation_rates(TS_data, time_points)
    else:
        alphas = [alpha] * ngenes

    # Get the indices of the candidate regulators
    idx = [i for i, gene in enumerate(gene_names) if gene in regulators]

    # Learn an ensemble of trees for each target gene, and compute scores for candidate regulators
    VIM = np.zeros((ngenes, ngenes))
    for i in range(ngenes):
        input_idx = idx.copy()
        if i in input_idx:
             input_idx.remove(i)
        vi = get_importances_single(TS_data, time_points, alphas[i], input_idx, i, SS_data, param)
        VIM[i, :] = vi

    time_end = time.time()
    print("Elapsed time: %.2f seconds" % (time_end - time_start))
    return VIM


def get_importances_single(TS_data, time_points, alpha, input_idx, output_idx, SS_data, param):
    h = 5  # define the value of time step
    ngenes = TS_data[0].shape[1]
    nexp = len(TS_data)
    nsamples_time = sum([expr_data.shape[0] for expr_data in TS_data])
    ninputs = len(input_idx)

    # Construct training sample
    input_matrix_time = np.zeros((0,ninputs))
    output_vect_time = np.zeros(0)

    nsamples_count = 0
    weight = [0.71,0.52,0.2,0.1]
    weight = [1, 1, 1, 1, 1, 1]
    for (i, current_timeseries) in enumerate(TS_data):
        current_time_points = time_points[i]
        npoints = current_timeseries.shape[0]
        time_diff_current = current_time_points[1:] - current_time_points[:npoints-1]
        current_timeseries_input=np.zeros((npoints-1,len(input_idx)))
        count = 0
        while count < h:
            current_timeseries_input[count:,] = current_timeseries_input[count:,] + weight[count]*current_timeseries[:npoints-1-count,input_idx]
            count = count + 1

        current_timeseries_output = (current_timeseries[1:,output_idx]-current_timeseries[:npoints-1,output_idx])/time_diff_current + alpha * current_timeseries[:npoints-1,output_idx]
        nsamples_current = current_timeseries_input.shape[0]
        #input_matrix_time[nsamples_count:nsamples_count + nsamples_current, :] = current_timeseries_input
        input_matrix_time = np.vstack([input_matrix_time,current_timeseries_input])
        output_vect_time = np.concatenate((output_vect_time,current_timeseries_output))
        #output_vect_time[nsamples_count:nsamples_count + nsamples_current] = current_timeseries_output
        nsamples_count += nsamples_current

    # Steady-state data
    if SS_data is not None:
        input_matrix_steady = SS_data[:, input_idx]
        output_vect_steady = SS_data[:, output_idx] * alpha

        # Concatenation
        input_all = np.vstack([input_matrix_steady, input_matrix_time])
        output_all = np.concatenate((output_vect_steady, output_vect_time))
    else:
        input_all = input_matrix_time
        output_all = output_vect_time


    treeEstimator = LGBMRegressor(**param)
    # Learn ensemble of trees
    treeEstimator.fit(input_all, output_all)
    # Compute importance scores
    feature_importances = treeEstimator.feature_importances_
    vi = np.zeros(ngenes)
    vi[input_idx] = feature_importances
    return vi
def get_scores(VIM, gold_edges, gene_names, regulators,index=0):

    idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
    pred_edges = [(gene_names[j], gene_names[i], score) for (i, j), score in np.ndenumerate(VIM) if i != j and j in idx]
    pred_edges = pd.DataFrame(pred_edges)
    pred_edges = pred_edges.iloc[:100000]
    final = pd.merge(pred_edges, gold_edges, on=[0, 1], how='inner')
    auroc = roc_auc_score(final['2_y'], final['2_x'])
    aupr = average_precision_score(final['2_y'], final['2_x'])
    fpr,tpr,tha = roc_curve(final['2_y'],final['2_x'])
    precision, recall, thresholds = precision_recall_curve(final['2_y'], final['2_x'])
    return auroc, aupr,precision,recall
        
        

