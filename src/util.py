import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
plt.rcParams["figure.dpi"] = 100


# ========================
# plotting functions

def plot_traj_comp(x_est, y_est, x_noise, y_noise, x, y, sample_idx, exp_name=''):

    plt.figure(dpi=100, figsize=(7, 4))

    # EKF State
    plt.plot(x_est, y_est, label='ekf_6states prediction', c='green', lw=2)

    # ground truth
    plt.plot(x, y, label='ground truth, 10hz', c='blue', lw=2, alpha=0.75)

    # Measurements
    plt.scatter(x_noise, y_noise, s=30, label='GPS noise, 10hz', marker='+', alpha=0.75, c='orange')

    # Start/Goal
    plt.scatter(x[0], y[0], s=75, label='start', c='g')
    plt.scatter(x[-1], y[-1], s=75, label='end', c='r')

    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title(f'sample [{sample_idx}]')
    plt.legend(loc='best')
    plt.show()

def plot_traj_comp_idx(ekf_results, sample_idx, exp_name=''):

    plt.figure(dpi=100, figsize=(7, 4))
    ekf_result = ekf_results[sample_idx]

    x_est = ekf_result['x_est'].values
    y_est = ekf_result['y_est'].values

    x = ekf_result['x'].values
    y = ekf_result['y'].values

    x_noise = ekf_result['x_noise'].values
    y_noise = ekf_result['y_noise'].values

    # EKF State
    plt.plot(x_est, y_est, label='ekf_6states prediction', c='green', lw=2)

    # ground truth
    plt.plot(x, y, label='ground truth, 10hz', c='blue', lw=2, alpha=0.75)

    # Measurements
    plt.scatter(x_noise, y_noise, s=30, label='GPS noise, 10hz', marker='+', alpha=0.75, c='orange')

    # Start/Goal
    plt.scatter(x[0], y[0], s=75, label='start', c='g')
    plt.scatter(x[-1], y[-1], s=75, label='end', c='r')

    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title(f'sample [{sample_idx}]: {exp_name}')
    plt.legend(loc='best')
    plt.show()

def plot_tot_traj_dist_err(traj_seg_pred, exp_name):

    length_m_lst = []
    length_m_pred_lst = []
    length_m_noise_lst = []

    for idx, df in enumerate(traj_seg_pred):
        length_m, length_m_pred, length_m_noise = traj_distances_proj(df)
        length_m_lst.append(length_m)
        length_m_pred_lst.append(length_m_pred)
        length_m_noise_lst.append(length_m_noise)
    
    plt.figure(figsize=((7, 4)), dpi=100)
    sns.histplot(np.array(length_m_lst) - np.array(length_m_pred_lst), binwidth=0.25, label='truth - pred')
    sns.histplot(np.array(length_m_lst) - np.array(length_m_noise_lst), binwidth=0.25, label='truth - noise', color='red')
    plt.title(f'ekf_6state: {exp_name}')
    plt.xlabel('total trajectory length error [m]')
    # plt.xlim(-15, 1)
    plt.ylim(0, 20)
    plt.legend()

def plot_proj_dist_err(pred_err, noise_err, exp_name):

    plt.figure(figsize=(7,4), dpi=100)
    ax = sns.histplot(
        noise_err,
        binwidth=0.02, label='spatial noise', color='red', kde=True)
    sns.histplot(
        pred_err, 
        binwidth=0.02, label='ekf_6states pred error', kde=True)

    ax.axvline(np.median(noise_err), color='k', lw=2, label='noise: median')
    ax.axvline(np.nanmedian(np.array(pred_err)), color='red', ls='--', lw=2, label='ekf_6states: median')

    plt.title(f'ekf_6states: {exp_name}')
    plt.xlabel('error distance [m]')
    # plt.ylim([0, 250])
    plt.xlim([-0.1, 1.2])
    plt.legend()
    plt.show()

def boxplot_proj_dist_err(pred_err, noise_err, exp_name):

    plt.figure(figsize=(7, 3), dpi=100)
    ax = sns.boxplot(data=[
        pred_err, noise_err
    ], orient='h')
    ax.set_xlim(0, 7)
    ax.set_yticks([0, 1], ['prediction error', 'noise error'])
    plt.title(f'ekf_6states boxplot: {exp_name}')
    plt.xlabel('coord euclidean error [m]')

# ========================
# error metrics

def traj_distances_proj(df):

    # trajectory distance after projection

    length_m = 0
    length_m_pred = 0
    length_m_noise = 0

    for index, row in df.iterrows():
        try:
            # actual
            start = (row['y'], row['x'])
            end = (df.iloc[index+1]['y'], df.iloc[index+1]['x'])
            length_m += math.dist(start, end)

            # predicted
            start = (row['y_est'], row['x_est'])
            end = (df.iloc[index+1]['y_est'], df.iloc[index+1]['x_est'])
            length_m_pred += math.dist(start, end)

            # noise
            start = (row['y_noise'], row['x_noise'])
            end = (df.iloc[index+1]['y_noise'], df.iloc[index+1]['x_noise'])
            length_m_noise += math.dist(start, end)

        except: # end of trajectory
            pass
    
    return length_m, length_m_pred, length_m_noise

def proj_dist_err(ekf_results):

    # euclidean distance error of each coordinate pair

    proj_dist_err_pred_lst = []
    proj_dist_err_noise_lst = []

    for seg_df in ekf_results:
        traj_est = np.array([seg_df['x_est'].values, seg_df['y_est'].values])
        traj_noise = np.array([seg_df['x_noise'].values, seg_df['y_noise'].values])
        traj_gt = np.array([seg_df['x'].values, seg_df['y'].values])

        for i in np.arange(seg_df.shape[0]):
            est_coord = (traj_est[0][i], traj_est[1][i])
            gt_noise = (traj_noise[0][i], traj_noise[1][i])
            gt_coord = (traj_gt[0][i], traj_gt[1][i])
            
            proj_dist_err_pred_lst.append(math.dist(est_coord, gt_coord))
            proj_dist_err_noise_lst.append(math.dist(gt_noise, gt_coord))
    
    return proj_dist_err_pred_lst, proj_dist_err_noise_lst