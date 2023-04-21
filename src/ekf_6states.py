import numpy as np
import pandas as pd
import pickle
import gzip
from sympy import Symbol, symbols, Matrix, sin, cos
from scipy.signal import savgol_filter
from util import *

sample_freq = 25 # 10hz
dt = 0.01 * sample_freq

v, theta, omega, dts, x, y, lat, lon, alpha = symbols('v theta \omega T x y lat lon alpha')

# transition matrix
A = Matrix([[x + (1 / omega**2) * ((v*omega + alpha * omega * dts) * sin(theta + omega * dts) + alpha * cos(theta + omega * dts) - v * omega * sin(theta) - alpha * cos(theta))],    
             [y + (1 / omega**2) * ((-v*omega - alpha * omega * dts) * cos(theta + omega * dts) + alpha * sin(theta + omega * dts) + v * omega * cos(theta) - alpha * sin(theta))],
             [omega + omega*dts],
             [alpha*dts + v],
             [omega],
             [alpha]])

# state vector         
state = Matrix([x, y, theta, v, omega, alpha])

# constants definition
max_Vx = 6.5 # m/s
max_alpha_Vx = 2.193 # m/s2
max_omega_Zv = 1.56 # deg/s
max_omega_Zv_accel = 0.5 # 1 # deg/s2

s_gps = 0.5 * max_alpha_Vx * dt ** 2
s_yaw = 0.001 * dt # max_omega_Zv * dt
s_vel = max_alpha_Vx * dt
s_omega = max_omega_Zv_accel * dt
s_accel = 0.25

sig_gps = 0.25
sig_theta = 0.1
sig_vel = 0 * dt
sig_omega = 0.01
sig_accel = 0

def load_data(data_ver='3'):

    # NOISE
    with gzip.open(f'../data/segment_noise_v{data_ver}/ll_seg_noise_v{data_ver}.pkl.gzip', 'rb') as f:
        ll_seg_noise = pickle.load(f)
    with gzip.open(f'../data/segment_noise_v{data_ver}/rl_seg_noise_v{data_ver}.pkl.gzip', 'rb') as f:
        rl_seg_noise = pickle.load(f)

    # GROUND TRUTH
    # with gzip.open(f'../data/segment_groud_truth_v2/rl_seg_gt_500_v2.pkl.gzip', 'rb') as f:
    #     ll_seg_gps_gt = pickle.load(f)
    # with gzip.open(f'../data/segment_groud_truth_v2/ll_seg_gt_500_v2.pkl.gzip', 'rb') as f:
    #     rl_seg_gps_gt = pickle.load(f)

    return ll_seg_noise, rl_seg_noise


def ekf_6_states(segment_df, param, filter_flag):

    # constants definition
    max_Vx = 6.5 # m/s
    max_alpha_Vx = 2.193 # m/s2
    max_omega_Zv = 1.56 # deg/s
    max_omega_Zv_accel = 0.5 # 1 # deg/s2

    s_gps = 0.5 * max_alpha_Vx * dt ** 2
    s_yaw = 0.001 * dt # max_omega_Zv * dt
    s_vel = max_alpha_Vx * dt
    s_omega = max_omega_Zv_accel * dt
    s_accel = 0.25

    sig_gps = 0.25
    sig_theta = 0.1
    sig_vel = 0 * dt
    sig_omega = 0.01
    sig_accel = 0

    x_noise = segment_df['x_noise'].values
    y_noise = segment_df['y_noise'].values
    Vx = segment_df['Vx'].values
    theta = segment_df['theta'].values/180.0*np.pi                                      # rad

    if filter_flag:
        omega_Zv = savgol_filter(segment_df['omega_Zv'].values/180.0*np.pi, 32, 4)      # rad/s
        alpha_Xv = savgol_filter(segment_df['alpha_Xv'].values, 32, 4)
    else:
        omega_Zv = segment_df['omega_Zv'].values/180.0*np.pi                            # rad/s
        alpha_Xv = segment_df['alpha_Xv'].values

    # set initial values
    x = np.matrix([
        x_noise[0], y_noise[0], theta[0], Vx[0], omega_Zv[0], alpha_Xv[0]
    ]).T

    observations = np.vstack((
        x_noise, y_noise, Vx, omega_Zv, alpha_Xv
    ))

    if param is None:
        P = 1 * np.eye(len(state))
        # process noise covariance Q matrix
        Q = np.diag([
            s_gps**2, s_gps**2, s_yaw**2, s_vel**2, s_omega**2, s_accel**2
        ])
        R = np.diag(
            [sig_gps**2, sig_gps**2, sig_vel**2, sig_omega**2, sig_accel**2]
        )
    elif len(param) == 8:
        # p_diag, q11, q33, q44, q55, q66, r44, r55

        P = param['p_diag'] * np.eye(len(state))
        Q = np.diag([
            param['q11'], param['q11'], param['q33'], param['q44'], param['q55'], param['q66']
        ])
        R = np.diag(
                [sig_gps**2, sig_gps**2, sig_vel**2, param['r44'], param['r55']]         
            )
    elif len(param) == 6:
        # q11, q22, q33, q44, q55, q66

        P = 5 * np.eye(len(state))
        Q = np.diag([
            param['q11'], param['q22'], param['q33'], param['q44'], param['q55'], param['q66']
        ])
        R = np.diag(
                [sig_gps**2, sig_gps**2, sig_vel**2, sig_omega**2, sig_accel**2]
            )
    else:
        P = param['p_diag'] * np.eye(len(state))
        Q = np.diag([
            param['q11'], param['q33'], param['q44'], param['q55'], param['q66']
        ])
        

    """ Q = np.diag([
        s_gps**2, s_gps**2, 0.0002, s_vel**2, 0.0002, s_accel**2
    ]) """

    """ Q = np.array([
        [s_gps**2, 0, 0.0001, 0.0001, 0.0001, 0.0001],
        [0, s_gps**2, 0.0001, 0.0001, 0.0001, 0.0001],
        [0, 0, s_yaw**2, 0.0001, 0.0001, 0.0001],
        [0, 0, 0.0001, 0.0001, s_vel**2, 0.0001],
        [0, 0, 0.0001, 0.0001, s_omega**2, 0],
        [0, 0, 0.0001, 0.0001, 0, s_accel**2],
    ]) """

    I = np.eye(len(state))

    # measurement function, H
    H = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ])

    m = observations.shape[1]
    nx = Q.shape[0]
    ny = R.shape[0]
    x_est = np.zeros((m, nx))      # prediction of state vector                                                       

    for filterstep in np.arange(m):
        
        if np.abs(omega_Zv[filterstep])<0.0001: # Driving straight
            x[4] = 0.0001

        if filterstep > 0:
            # state transition
            x[0] = x[0] + (1 / x[4]**2) * ((x[3]*x[4] + x[5] * x[4] * dt) * \
                np.sin(x[2] + x[4]* dt) + x[5] * np.cos(x[2] + x[4] * dt) - x[3] *  \
                x[4] * np.sin(x[2]) - x[5] * np.cos(x[2]))
            x[1] = x[1] + (1 / x[4]**2) * ((-x[3]*x[4] - x[5] * x[4] * dt) * \
                np.cos(x[2] + x[4]* dt) + x[5] * np.sin(x[2] + x[4] * dt) + x[3] * \
                x[4] * np.cos(x[2]) - x[5] * np.sin(x[2]))
            x[2] = (x[2] + x[4] * dt + np.pi) % (2.0 * np.pi) - np.pi
            x[3] = x[3] + x[5] * dt 
            x[4] = x[4]
            x[5] = x[5]
        
        
        # Calculate the Jacobian of the Dynamic Matrix A
        a13 = ((-x[4]*x[3]*np.cos(x[2]) + x[5]*np.sin(x[2]) - x[5]*np.sin(dt*x[4] + x[2]) + \
            (dt*x[4]*x[5] + x[4]*x[3])*np.cos(dt*x[4] + x[2]))/x[4]**2).item(0)

        a14 = ((-x[4]*np.sin(x[2]) + x[4]*np.sin(dt*x[4] + x[2]))/x[4]**2).item(0)

        a15 = ((-dt*x[5]*np.sin(dt*x[4] + x[2]) + dt*(dt*x[4]*x[5] + x[4]*x[3])* \
            np.cos(dt*x[4] + x[2]) - x[3]*np.sin(x[2]) + (dt*x[5] + x[3])* \
            np.sin(dt*x[4] + x[2]))/x[4]**2 - 2*(-x[4]*x[3]*np.sin(x[2]) - x[5]* \
            np.cos(x[2]) + x[5]*np.cos(dt*x[4] + x[2]) + (dt*x[4]*x[5] + x[4]*x[3])* \
            np.sin(dt*x[4] + x[2]))/x[4]**3).item(0)

        a16 = ((dt*x[4]*np.sin(dt*x[4] + x[2]) - np.cos(x[2]) + np.cos(dt * x[4] + x[2]))/x[4]**2).item(0)

        a23 = ((-x[4] * x[3] * np.sin(x[2]) - x[5] * np.cos(x[2]) + x[5] * np.cos(dt * x[4] + x[2]) - \
            (-dt * x[4]*x[5] - x[4] * x[3]) * np.sin(dt * x[4] + x[2])) / x[4]**2).item(0)
        a24 = ((x[4] * np.cos(x[2]) - x[4]*np.cos(dt*x[4] + x[2]))/x[4]**2).item(0)
        a25 = ((dt * x[5]*np.cos(dt*x[4] + x[2]) - dt * (-dt*x[4]*x[5] - x[4] * x[3]) * \
            np.sin(dt * x[4] + x[2]) + x[3]*np.cos(x[2]) + (-dt*x[5] - x[3])*np.cos(dt*x[4] + x[2]))/ \
            x[4]**2 - 2*(x[4]*x[3]*np.cos(x[2]) - x[5] * np.sin(x[2]) + x[5] * np.sin(dt*x[4] + x[2]) + \
            (-dt * x[4] * x[5] - x[4] * x[3])*np.cos(dt*x[4] + x[2]))/x[4]**3).item(0)
        a26 =  ((-dt*x[4]*np.cos(dt*x[4] + x[2]) - np.sin(x[2]) + np.sin(dt*x[4] + x[2]))/x[4]**2).item(0)
            
        JA = np.matrix([[1.0, 0.0, a13, a14, a15, a16],
                        [0.0, 1.0, a23, a24, a25, a26],
                        [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        
        
        # Project the error covariance ahead
        P = JA * P * JA.T + Q
        
        # Measurement Update (Correction)
        # ===============================
        # Measurement Function
        hx = np.matrix([[float(x[0])],
                        [float(x[1])],
                        [float(x[3])],
                        [float(x[4])],
                        [float(x[5])]])        
        
        S = H * P * H.T + R
        K = (P * H.T) * np.linalg.inv(S)

        # Update the estimate via
        Z = observations[:, filterstep].reshape(H.shape[0], 1)
        y = Z - (hx) # innovation or residual
        x = x + (K * y) # update estimated state

        # Update the error covariance
        P = (I - (K * H)) * P

        x_est[filterstep] = x.T

    return x_est, x_noise, y_noise, segment_df['x'].values, segment_df['y'].values

# ========================
# batch evaluation functions

def ekf_batch_eval(batch_df, param, filter_flag=False):

    seg_pred_batch = []

    for df_idx, df in enumerate(batch_df):
        try:
            x_est_sample, x_noise, y_noise, x, y = ekf_6_states(df, param, filter_flag)
            x_est_df = pd.DataFrame({
                'x_est': x_est_sample[:, 0],
                'y_est': x_est_sample[:, 1],
                'theta': x_est_sample[:, 2],
                'Vx_est': x_est_sample[:, 3],
                'x_noise': x_noise,
                'y_noise': y_noise,
                'x': x,
                'y': y
                }
            )
            seg_pred_batch.append(x_est_df)
        except:
            print(df_idx)
    
    return seg_pred_batch

def main():
    return None

if __name__ == "__main__":
    main()

