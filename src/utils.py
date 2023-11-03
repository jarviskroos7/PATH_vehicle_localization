import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
import pandas as pd
import pyproj

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

def plot_tot_traj_dist_err(traj_seg_pred, exp_name, ylim=20, binwidth=0.25):

    length_m_lst = []
    length_m_pred_lst = []
    length_m_noise_lst = []

    for idx, df in enumerate(traj_seg_pred):
        length_m, length_m_pred, length_m_noise = traj_distances_proj_vectorized(df)
        length_m_lst.append(length_m)
        length_m_pred_lst.append(length_m_pred)
        length_m_noise_lst.append(length_m_noise)
    
    delta_pred = np.array(length_m_lst) - np.array(length_m_pred_lst)
    delta_noise = np.array(length_m_lst) - np.array(length_m_noise_lst)

    print('delta_pred stats:')
    print(pd.Series(delta_pred).describe())
    print()
    print('delta_noise stats:')
    print(pd.Series(delta_noise).describe())

    plt.figure(figsize=((7, 4)), dpi=100)
    sns.histplot(delta_pred, binwidth=binwidth, label='truth - pred')
    sns.histplot(delta_noise, binwidth=binwidth, label='truth - noise', color='red')
    plt.title(f'ekf_6state: {exp_name}')
    plt.xlabel('total trajectory length error [m]')
    # plt.xlim(-15, 1)
    plt.ylim(0, ylim)
    plt.legend()
    plt.show()

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
    # ax.set_xlim(0, 7)
    ax.set_yticks([0, 1], ['prediction error', 'noise error'])
    plt.title(f'ekf_6states boxplot: {exp_name}')
    plt.xlabel('coord euclidean error [m]')
    plt.show()

# ========================
# error metrics

def traj_distances_proj_vectorized(df):

    """
    vectorization applied
    trajectory distance after projection
    """

    x = df['x'].values
    y = df['y'].values
    x_noise = df['x_noise'].values
    y_noise = df['y_noise'].values
    x_est = df['x_est'].values
    y_est = df['y_est'].values

    def distance(x_start, y_start, x_end, y_end):
        start = (x_start, y_start)
        end = (x_end, y_end)
        return math.dist(start, end)
    
    length_gt = np.sum(
        np.vectorize(distance)(
            x[:-1], y[:-1], x[1:], y[1:]
        )
    )

    length_pred = np.sum(
        np.vectorize(distance)(
            x_est[:-1], y_est[:-1], x_est[1:], y_est[1:]
        )
    )

    length_noise = np.sum(
        np.vectorize(distance)(
            x_noise[:-1], y_noise[:-1], x_noise[1:], y_noise[1:]
        )
    )

    return length_gt, length_pred, length_noise

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


def next_coord_conversion(
        coord1, coord2, coord_prev=[0, 0]
    ) -> list:

    # normalize GPS coordinate in relation to a given coord_prev

    x0 = coord_prev[0]
    y0 = coord_prev[1]
    lat1 = coord1[0] # x
    lon1 = coord1[1] # y
    lat2 = coord2[0]
    lon2 = coord2[1]

    geodesic = pyproj.Geod(ellps='WGS84')
    bearing, inv_bearing, distance = geodesic.inv(lon1, lat1, lon2, lat2)

    coord_next = [
        x0 + distance * math.cos(bearing * math.pi / 180), 
        y0 + distance * math.sin(bearing * math.pi / 180)
    ]

    return coord_next

def normalize_lane_gps(lane_df, col=['x', 'y']) -> pd.DataFrame:

    # normalize the GPS coordinates of the given lane_df to a planer axis
    # with (0, 0) origin

    lane_coord_next_lst = []
    x = col[0]
    y = col[1]

    for index, row in lane_df.iterrows():
    
        coord1 = [row[y], row[x]]
        
        if index == 0:
            coord2 = [lane_df.iloc[index+1][y], lane_df.iloc[index+1][x]]
            coord_next = next_coord_conversion(coord1, coord2)
        elif index == lane_df.shape[0]-1:
            pass
        else:
            coord2 = [lane_df.iloc[index+1][y], lane_df.iloc[index+1][x]]
            coord_prev = lane_coord_next_lst[-1]
            coord_next = next_coord_conversion(coord1, coord2, coord_prev)
        
        lane_coord_next_lst.append(coord_next)
        
    lane_coord_next_lst = np.asarray(lane_coord_next_lst)
    
    # shift so that the minimum coordinates are at the origin
    lane_coord_next_lst[:, 0] = lane_coord_next_lst[:, 0] - min(lane_coord_next_lst[:, 0])
    lane_coord_next_lst[:, 1] = lane_coord_next_lst[:, 1] - min(lane_coord_next_lst[:, 1])

    lane_df[y] = lane_coord_next_lst[:, 0]
    lane_df[x] = lane_coord_next_lst[:, 1]

    return lane_df

def sample_gps(traj_df, sampling_rate) -> pd.DataFrame:
    """
    down sample the given dataframe to the sampling_rate 
    """

    rtk_sampling_rate = 100 # [hz]
    sampling_spacing = round(rtk_sampling_rate / sampling_rate)
    traj_df_sampled = traj_df.iloc[::sampling_spacing]

    return traj_df_sampled

def gps_noise_augmentation(traj_df, noise_perc, colns, seed=0) -> pd.DataFrame:

    np.random.seed(seed)
    seeds = np.random.randint(10, size=len(colns))
    traj_df_copy = traj_df.copy()
    std = 0.5

    for idx in np.arange(len(colns)):
        coln = colns[idx]
        np.random.seed(seeds[idx])
        traj_df_copy[f'{coln}_noise'] = traj_df_copy[coln] + noise_perc * std * np.random.randn(traj_df_copy.shape[0])
    
    print('Random seeds =', seeds)
    return traj_df_copy

def seg_generation(track_df, seg_size_avg, seg_size_std, num_samples=500, seed=0, ver=1) -> list:
    """
    sample continuous trajectory segments from full track with length from a normal distribution

    return: list of pd.Dataframe
    """
    
    # ensure reproducibility
    np.random.seed(seed)
    seg = []

    if ver == 1:
        # random start, random length
        for length in np.random.normal(seg_size_avg, seg_size_std, num_samples).round():
            
            start_idx = np.random.randint(track_df.shape[0]-length)
            length = int(length)
            seg.append(track_df.iloc[start_idx:start_idx+length])
    elif ver == 2:
        # fixed start, random length
        start_idx = 0 # start at the begining of the track
        for length in np.random.randint(track_df.shape[0], size=num_samples):

            seg.append(track_df.iloc[start_idx:start_idx+length])
        
    return seg