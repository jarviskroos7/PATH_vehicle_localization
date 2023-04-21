from tqdm import tqdm
from hyperopt import fmin, tpe, hp, anneal, Trials
import numpy as np
import sys
from ekf_6states import *
from util import *
import logging
import warnings
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

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
    # 'p_diag': hp.choice('p_diag', np.linspace(0, 10, 20)),
    'q11': hp.normal('q11', s_gps**2, 0.001),
    'q22': hp.normal('q22', s_gps**2, 0.001),
    'q33': hp.normal('q33', s_yaw**2, 2.5e-8),
    'q44': hp.normal('q44', s_vel**2, 0.1),
    'q55': hp.normal('q55', s_omega**2, 0.005),
    'q66': hp.normal('q66', s_accel**2, 0.02),
    }


# ========================
# loss functions

# sigma1: coordinate-wise euclidean error
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

# sigma2: Segment-wise total length error
def batch_tot_dist(space):

    # ekf_results = ekf_batch_eval(batch_df=sample_trajs, param=space) # subsample
    ekf_results = ekf_batch_eval(batch_df=rl_seg_noise, param=space) # full sample
    total_traj_err_lst = []

    for seg_df in ekf_results:
        
        # total trajectory length
        traj_length, traj_length_pred, _ = traj_distances_proj_vectorized(seg_df)
        total_traj_err_lst.append(abs(traj_length - traj_length_pred))
    
    return np.average(total_traj_err_lst)

def batch_tot_dist_post(ekf_results):

    total_traj_err_lst = []

    for seg_df in ekf_results:
        traj_length, traj_length_pred, _ = traj_distances_proj_vectorized(seg_df)
        total_traj_err_lst.append(abs(traj_length - traj_length_pred))
    
    return np.average(total_traj_err_lst)

# sigma3: Coordinate-wise RMSE
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

# sigma4: combined normalized segment-wise total trajectory length and RMSE
def batch_tot_dist_and_rmse(space):
    
    # partial sample
    # ekf_results = ekf_batch_eval(batch_df=sample_trajs, param=space)
    # full sample
    ll_seg_noise, rl_seg_noise = load_data(data_ver='5')
    ekf_results = ekf_batch_eval(batch_df=rl_seg_noise, param=space)

    rmse_lst = []
    total_traj_err_lst = []

    for seg_df in ekf_results:
        # coordinate-wise
        traj_est = np.array([seg_df['x_est'].values, seg_df['y_est'].values])
        traj_gt = np.array([seg_df['x'].values, seg_df['y'].values])
        # total length
        traj_length, traj_length_pred, _ = traj_distances_proj_vectorized(seg_df)

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
        traj_length, traj_length_pred, _ = traj_distances_proj_vectorized(seg_df)

        rmse_lst.append(np.sqrt(mean_squared_error(traj_est, traj_gt)))
        total_traj_err_lst.append(abs(traj_length - traj_length_pred))
    
    # normalize both error metrics
    rmse_lst = normalize(np.array([rmse_lst]))[0]
    total_traj_err_lst = normalize(np.array([total_traj_err_lst]))[0]
    comb_err_lst = rmse_lst + total_traj_err_lst

    return np.average(np.array(comb_err_lst))

def load_exp_param(experiment):
    with open(f'../models/hyperopt_trials/{experiment}.pkl', 'rb') as f:
        exp_trial = pickle.load(f)
    exp_best = exp_trial.trials[0]['misc']['vals']
    for key in exp_best.keys():
        exp_best[key] = exp_best[key][0]

    return exp_best

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


def main(mode, experiment, space, error_metric=None):

    if mode == 'train':
        exp_trial, exp_best = hypeopt_tune(space, error_metric, iterations=50)
        
        for param in exp_best.keys():
            print(exp_best[param])

        pickle.dump(exp_trial, open(f"../models/hyperopt_trials/{experiment}.pkl", "wb"))
    
    else:
        exp_best = load_exp_param(experiment)
    
    logger.info('===== batch estimation... =====')

    batch_pred = ekf_batch_eval(rl_seg_noise, exp_best)
    logger.info(f'{experiment} error stats:')
    logger.info('')
    logger.info(f'avg total segment length error, {round(batch_tot_dist_post(batch_pred), 4)} [m]')
    logger.info(f'avg RMSE, {round(batch_rmse_post(batch_pred), 4)} [m]')

    plot_tot_traj_dist_err(batch_pred, exp_name=experiment, ylim=100, binwidth=2)
    exp_proj_dist_err_pred, exp_proj_dist_err_noise = proj_dist_err(batch_pred)
    boxplot_proj_dist_err(exp_proj_dist_err_pred, exp_proj_dist_err_noise, exp_name=experiment)
    plot_proj_dist_err(exp_proj_dist_err_pred, exp_proj_dist_err_noise, exp_name=experiment)

    return None

if __name__ == "__main__":
    
    mode = sys.argv[1]                  # "train", "predict"
    experiment = sys.argv[2]            # experiment name, ex. "exp8_sigma2_v3_50_trials"
    data_version = sys.argv[3]          # "3", "5"
    error_metric = int(sys.argv[4])     # "0, 1, 2, 3, 4"

    ll_seg_noise, rl_seg_noise = load_data(data_version)
    error_metric = [
        None, batch_gc_dist, batch_tot_dist, batch_rmse, batch_tot_dist_and_rmse
        ][error_metric-1]

    if mode == "predict":
        exp_trial = load_exp_param(experiment)
    else:
        exp_trial = None
        
    
    main(mode, experiment, space_v3, error_metric)