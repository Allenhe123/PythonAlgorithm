
##################################
# 估计轨迹
# 基本的卡尔曼滤波只能对线性的轨迹进行滤波！！！
# 非线性的轨迹需要用扩展卡尔曼滤波（EKF）
##################################

import matplotlib.pyplot as plt
import math
import random
import numpy as np

t = 15        # 仿真时间
Ts = 0.1      # 采样周期
length = t / Ts  # 仿真步数

# 增加加速度
X = np.zeros((int(length), 6))
X[0,:] = np.array([0, 0, 1, 1, 0, 0]) # 状态模拟的初值(x, y, vx, vy, ax, ay)

Z = np.zeros((int(length), 2))

for i in range(1, int(length)):
    x0 = X[i - 1, 0]
    y0 = X[i - 1, 1]
    vx0 = X[i - 1, 2]
    vy0 = X[i - 1, 3]
    ax0 = X[i - 1, 4]
    ay0 = X[i - 1, 5]
    x1 = x0 + vx0 * Ts + 0.5 * ax0 * Ts**2
    y1 = y0 + vy0 * Ts + 0.5 * ay0 * Ts**2
    vx1 = vx0 + ax0 * Ts
    vy1 = vy0 + ay0 * Ts
    ax1 = ax0 + (random.random() - 0.5) * 2
    ay1 = ay0 + (random.random() - 0.5) * 2
    X[i,:] = [x1, y1, vx1, vy1, ax1, ay1]

for k in range(0, int(length)):
    x = X[k, 0] +  5 * (random.random() - 0.5)
    y = X[k, 1] +  5 * (random.random() - 0.5)
    Z[k:] = [x, y]

################################ ekf filter
observed_x = Z[:,0]
observed_y = Z[:,1]


ax = []
ay = []
bx = []
by = []
distances = []

def kalman(x, P, measurement, R, Q, F, H, B, u):
    '''
    Parameters:
    x: initial state
    P: initial uncertainty convariance matrix
    measurement: observed position (same shape as H*x)
    R: measurement noise (same shape as H)
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    F: next state function: x_prime = F*x
    H: measurement function: position = H*x

    Return: the updated and predicted new values for (x, P)

    See also http://en.wikipedia.org/wiki/Kalman_filter

    This version of kalman can be applied to many different situations by
    appropriately defining F and H

    这里的状态是（坐标x， 坐标y， 速度x， 速度y），观察值是（坐标x， 坐标y），所以H = eye(2, 4)

    '''
    # UPDATE x, P based on measurement m
    # distance between measured and current position-belief

    x = F * x + B * u
    P = F * P * F.T + Q

    ax.append(x[0,0])
    ay.append(x[1,0])
    print('(',x[0,0], ',', x[1,0], ')')

    # 更新卡尔曼增益
    S = H * P * H.T + R  # 计算卡尔曼增益时候的分母
    K = P * H.T * S.I    # 卡尔曼增益

    # 更新估计值
    y = np.matrix(measurement).T - H * x
    x = x + K * y

    bx.append(x[0,0])
    by.append(x[1,0])
    print('(', x[0,0], ',', x[1,0], ')')
    print(' ')
    print(' ')

    # 更新估计值的方差
    # P = P - K * H * P
    I = np.matrix(np.eye(F.shape[0])) # identity matrix
    P = (I - K * H) * P

    return x, P

def demo_kalman_xy():
    x = np.matrix('0 0 0 0 0 0').T                            #初始化估计值（x,y,vx,vy）
    x[0] = observed_x[0]
    x[1] = observed_y[0]
    x[2] = 1
    x[3] = 1

    F = np.matrix('1 0 0.1 0 0.005 0; 0 1 0 0.1 0 0.005; 0 0 1 0 0.1 0; 0 0 0 1 0 0.1; 0 0 0 0 1 0; 0 0 0 0 0 1') # 状态转移矩阵，方阵，将k-1与k时刻的状态关联起来
    H = np.matrix('1 0 0.1 0 0.005 0; 0 1 0 0.1 0 0.005')                         # 观测矩阵H, z = H * x
    B = 0                                                     # B是可选的控制输入 的增益，在大多数实际情况下并没有控制增益，所以 这一项很愉快的变成零了
    u = np.matrix('0 0 0 0 0 0').T                            # 外部输入为0
    P = np.matrix(np.eye(6)) * 0.1                            # 初始化估计值方差
    Q = np.matrix(np.eye(6)) * 0.001                          # 过程噪声方差 -- 预测噪声协方差矩阵Q
    R = np.eye(2) * 0.2                                       # 观测噪声协方差矩阵R：假设观测过程上存在一个高斯噪声，协方差矩阵为R

    result = []
    for meas in zip(observed_x, observed_y):
        x, P = kalman(x, P, meas, R,  Q, F, H, B, u)
        result.append((x[:2]).tolist())

    kalman_x, kalman_y = zip(*result)

    # plt.plot(X[:, 0], X[:, 1])
    plt.plot(observed_x, observed_y, 'bo')
    plt.plot(kalman_x, kalman_y, color = 'r')
    plt.show()

    idx = []
    for i in range(len(ax)):
        idx.append(i)
        dis = math.sqrt( (ax[i] - bx[i]) * (ax[i] - bx[i]) + (ay[i] - by[i]) * (ay[i] - by[i])  )
        distances.append(dis)

    plt.figure()
    plt.plot(ax, ay, color = 'g')
    plt.plot(bx, by, color = 'r')
    plt.show()

    plt.figure()
    plt.plot(idx, distances, color='g')
    plt.show()

demo_kalman_xy()