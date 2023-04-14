from tqdm import tqdm
from hyperopt import fmin, tpe, hp, anneal, Trials
import pandas as pd
import numpy as np
import sys
from ekf_6states import *
from util import *
import logging

from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error

# ========================
# custom search spaces

space_v1 = {
        'p_diag': hp.uniform('p_diag', 0, 10),
        'q11': hp.uniform('q11', 0, 0.005),
        'q33': hp.uniform('q33', 0, 0.2),
        'q44': hp.uniform('q44', 0, 0.3),
        'q55': hp.uniform('q55', 0, 0.1),
        'q66': hp.uniform('q66', 0, 0.2)
    }

space_v2 = {
    'p_diag': hp.uniform('p_diag', 0, 10),
    'q11': hp.uniform('q11', 0, 0.005),
    'q33': hp.uniform('q33', 0, 0.2),
    'q44': hp.uniform('q44', 0, 0.3),
    'q55': hp.uniform('q55', 0, 0.1),
    'q66': hp.uniform('q66', 0, 0.2),
    'r44': hp.uniform('r44', 0, 1),
    'r55': hp.uniform('r55', 0, 1),
}

space_v3 = {
    'p_diag': hp.choice('p_diag', np.linspace(0, 10, 20)),
    'q11': hp.loguniform('q11', 0, 0.005),
    'q33': hp.loguniform('q33', 0, 0.1),
    'q44': hp.loguniform('q44', 0, 0.3),
    'q55': hp.loguniform('q55', 0, 0.2),
    'q66': hp.loguniform('q66', 0, 0.1),
    'r44': hp.uniform('r44', 0, 2),
    'r55': hp.uniform('r55', 0, 2),
}


# ========================
# loss functions

# exp1: coordinate-wise euclidean error
def batch_gc_dist(space):

    # ekf_results = ekf_batch_eval(batch_df=sample_trajs, param=space) # subsample
    ekf_results = ekf_batch_eval(batch_df=rl_seg_noise, param=space) # full sample
    gc_err_lst = []

    for seg_df in ekf_results:
        traj_est = np.array([seg_df['x_est'].values, seg_df['y_est'].values])
        traj_gt = np.array([seg_df['x'].values, seg_df['y'].values])
        traj_gc_err_lst = []

        for i in np.arange(len(seg_df)):
            est_coord = (traj_est[0][i], traj_est[1][i]) 
            gt_coord = (traj_gt[0][i], traj_gt[1][i])
            traj_gc_err_lst.append(math.dist(est_coord, gt_coord))

        gc_err_lst.append(sum(traj_gc_err_lst))
    
    return np.average(gc_err_lst)


def batch_gc_dist_post(ekf_results):

    gc_err_lst = []

    for seg_df in ekf_results:
        traj_est = np.array([seg_df['x_est'].values, seg_df['y_est'].values])
        traj_gt = np.array([seg_df['x'].values, seg_df['y'].values])
        traj_gc_err_lst = []

        for i in np.arange(len(seg_df)):
            est_coord = (traj_est[0][i], traj_est[1][i]) 
            gt_coord = (traj_gt[0][i], traj_gt[1][i])
            traj_gc_err_lst.append(math.dist(est_coord, gt_coord))

        gc_err_lst.append(sum(traj_gc_err_lst))
    
    return np.average(gc_err_lst)

# exp2: Segment-wise total length error
def batch_tot_dist(space):

    # ekf_results = ekf_batch_eval(batch_df=sample_trajs, param=space) # subsample
    ekf_results = ekf_batch_eval(batch_df=rl_seg_noise, param=space) # full sample
    total_traj_err_lst = []

    for seg_df in ekf_results:
        
        # total trajectory length
        traj_length, traj_length_pred, _ = traj_distances_proj(seg_df)
        total_traj_err_lst.append(abs(traj_length - traj_length_pred))
    
    return np.average(total_traj_err_lst)

def batch_tot_dist_post(ekf_results):

    total_traj_err_lst = []

    for seg_df in ekf_results:
        traj_length, traj_length_pred, _ = traj_distances_proj(seg_df)
        total_traj_err_lst.append(abs(traj_length - traj_length_pred))
    
    return np.average(total_traj_err_lst)

# exp3: Coordinate-wise RMSE
def batch_rmse(space):
    
    # ekf_results = ekf_batch_eval(batch_df=sample_trajs, param=space) # subsample
    ekf_results = ekf_batch_eval(batch_df=rl_seg_noise, param=space) # full sample
    rmse_lst = []

    for seg_df in ekf_results:
        traj_est = np.array([seg_df['x_est'].values, seg_df['y_est'].values])
        traj_gt = np.array([seg_df['x'].values, seg_df['y'].values])

        rmse_lst.append(np.sqrt(mean_squared_error(traj_est, traj_gt)))
    
    return np.average(rmse_lst)

def batch_rmse_post(ekf_results):

    rmse_lst = []

    for seg_df in ekf_results:
        traj_est = np.array([seg_df['x_est'].values, seg_df['y_est'].values])
        traj_gt = np.array([seg_df['x'].values, seg_df['y'].values])

        rmse_lst.append(np.sqrt(mean_squared_error(traj_est, traj_gt)))
    
    return np.average(rmse_lst)

# exp4: combined normalized segment-wise total trajectory length and RMSE
def batch_tot_dist_and_rmse(space):
    
    # partial sample
    # ekf_results = ekf_batch_eval(batch_df=sample_trajs, param=space)
    # full sample
    ekf_results = ekf_batch_eval(batch_df=rl_seg_noise, param=space)

    rmse_lst = []
    total_traj_err_lst = []

    for seg_df in ekf_results:
        # coordinate-wise
        traj_est = np.array([seg_df['x_est'].values, seg_df['y_est'].values])
        traj_gt = np.array([seg_df['x'].values, seg_df['y'].values])
        # total length
        traj_length, traj_length_pred, _ = traj_distances_proj(seg_df)

        rmse_lst.append(np.sqrt(mean_squared_error(traj_est, traj_gt)))
        total_traj_err_lst.append(abs(traj_length - traj_length_pred))
    
    # normalize both error metrics
    rmse_lst = normalize(np.array([rmse_lst]))[0]
    total_traj_err_lst = normalize(np.array([total_traj_err_lst]))[0]
    comb_err_lst = rmse_lst + total_traj_err_lst

    return np.average(np.array(comb_err_lst))


def batch_tot_dist_and_rmse_post(ekf_results):

    rmse_lst = []
    total_traj_err_lst = []

    for seg_df in ekf_results:
        # coordinate-wise
        traj_est = np.array([seg_df['x_est'].values, seg_df['y_est'].values])
        traj_gt = np.array([seg_df['x'].values, seg_df['y'].values])
        # total length
        traj_length, traj_length_pred, _ = traj_distances_proj(seg_df)

        rmse_lst.append(np.sqrt(mean_squared_error(traj_est, traj_gt)))
        total_traj_err_lst.append(abs(traj_length - traj_length_pred))
    
    # normalize both error metrics
    rmse_lst = normalize(np.array([rmse_lst]))[0]
    total_traj_err_lst = normalize(np.array([total_traj_err_lst]))[0]
    comb_err_lst = rmse_lst + total_traj_err_lst

    return np.average(np.array(comb_err_lst))

# ========================
# hyperopt tuning function

def hypeopt_tune(space, err_func, iterations):

    trials = Trials()
    best = fmin(
        fn=err_func,                        # function to optimize
        space=space,                        # search space 
        algo=tpe.suggest,                   # optimization algorithm, hyperotp will select its parameters automatically
        max_evals=iterations,               # maximum number of iterations
        trials=trials,                      # logging
        rstate=np.random.default_rng(12)    # fixing random state for the reproducibility
    )

    return trials, best


def main(mode):

    


    return None

if __name__ == "__main__":
    
    mode = sys.argv[1]
    exp = int(sys.argv[2])
    sample_size = int(sys.argv[3])

    if exp <= 6:
        exp_trial = load_param(exp, sample_size)
    else:
        exp_trial = None
        
    ll_seg_noise, rl_seg_noise, ll_seg_gps_gt, rl_seg_gps_gt = load_data()

    main(mode, )