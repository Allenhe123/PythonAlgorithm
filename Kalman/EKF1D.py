
##################################
# 估计轨迹
# 基本的卡尔曼滤波只能对线性的轨迹进行滤波
# 非线性的轨迹需要用扩展卡尔曼滤波（EKF）
##################################

import matplotlib.pyplot as plt
import math
import random
import numpy as np
import os



############################################################

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

    # 更新卡尔曼增益
    S = H * P * H.T + R  # 计算卡尔曼增益时候的分母
    K = P * H.T * S.I    # 卡尔曼增益

    # 更新估计值
    y = np.matrix(measurement).T - H * x
    x = x + K * y

    # 更新估计值的方差
    # P = P - K * H * P
    I = np.matrix(np.eye(F.shape[0])) # identity matrix
    P = (I - K * H) * P

    return x, P


def readWave2(rid):
    file = os.path.join(os.getcwd(), 'files\\wave.txt')
    result = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            lst = line.split(' ')
            if lst[1] == rid:
                result.append(line)

    strlst = result[0].split(' ')
    id = int(strlst[0])
    starttime = float(strlst[2])

    listT = []
    listP = []
    mean = 0
    cnt = 0
    for l in result:
        lst = l.split(' ')
        listT.append(float(lst[2]))
        listP.append(float(lst[3]))
        mean += float(lst[3])
        cnt += 1
        if int(lst[0]) >= 399:
            break

    # print(listT[len(listT) - 1])
    deltaT = listT[len(listT) - 1] - listT[0]
    mean /= cnt
    # print(deltaT)
    # print(listT[1] - listT[0])
    # print(listP[0])
    # print(mean)
    # print(min(listP), max(listP))

    return listT, listP

def demo_kalman_xy():
    t = 13.46587                  # 仿真时间
    Ts = 0.03375                  # 采样周期
    steps = int(t / Ts)           # 仿真步数

    T = []
    X = []
    Noise = []
    Z = []

    for i in range(steps):
        T.append(2 * math.pi * i * Ts)

    for i in range(steps):
        a = math.sin(2 * math.pi * i * Ts)
        X.append(a)

    for i in range(steps):
        e = 2 * (random.random() - 0.5)
        Noise.append(e)

        b = X[i] +  e
        Z.append(b)

    print(T)
    print(X)
    print(Z)

    A = 1                                                     # 状态转移矩阵，将k-1与k时刻的状态关联起来
    AT = 1
    H = 1                                                     # 观测矩阵H, z = H * x
    HT = 1
    B = 0                                                     # B是可选的控制输入 的增益，在大多数实际情况下并没有控制增益，所以 这一项很愉快的变成零了
    u = 0                                                     # 外部输入为0
    P = 0.1                                                   # 初始化估计值方差 -- 误差矩阵
    Q = 0.001                                                 # 预测噪声协方差矩阵Q -- 过程噪声方差

    narray1 = np.array(Noise)
    sum1 = narray1.sum()
    narray2 = narray1 * narray1
    sum2 = narray2.sum()
    mean = sum1 / len(Noise)
    var = sum2 / len(Noise) - mean ** 2
    R = var                                                   # 观测噪声协方差矩阵R：假设观测过程上存在一个高斯噪声，协方差矩阵为R

    XX = []
    for i in range(steps):
        XX.append(0)                                         # 初始化估计值

    for k in range(1, len(X)):
        # 预测
        xx = XX[k-1] + math.cos(T[k - 1]) * 2 * math.pi * Ts
        PP = A * P * AT + Q

        # 校正
        K = PP * HT / (H * PP * HT + R)
        XX[k] = xx + K * (Z[k] - H * xx)
        P = (1 - K * H) * PP

    plt.figure()
    plt.plot(T, X, color='r', label='ori')
    plt.plot(T, Z, color='g', label='noise')
    plt.plot(T, XX, color='b', label='kalman')
    plt.show()

def wave_kalman_xy():
    T, Z = readWave2('31401')
    print(T)
    print(Z)
    steps = len(T)
    print(steps)
    Ts = (T[1] - T[0])
    print(steps)
    print(Ts)

    A = 1                                                     # 状态转移矩阵，将k-1与k时刻的状态关联起来
    AT = 1
    H = 1                                                     # 观测矩阵H, z = H * x
    HT = 1
    B = 0                                                     # B是可选的控制输入 的增益，在大多数实际情况下并没有控制增益，所以 这一项很愉快的变成零了
    u = 0                                                     # 外部输入为0
    P = 0.1                                                   # 初始化估计值方差 -- 误差矩阵
    Q = 0.001                                                 # 预测噪声协方差矩阵Q -- 过程噪声方差
    R = 0.001                                                  # 观测噪声协方差矩阵R：假设观测过程上存在一个高斯噪声，协方差矩阵为R

    XX = []
    for i in range(steps):
        XX.append(0)                                         # 初始化估计值

    for k in range(1, steps):
        # 预测
        # + math.cos(T[k - 1]) * k * Ts
        xx = XX[k-1]
        PP = A * P * AT + Q

        # 校正
        K = PP * HT / (H * PP * HT + R)
        XX[k] = xx + K * (Z[k] - H * xx)
        P = (1 - K * H) * PP

    plt.figure()
    # plt.plot(T, Z, color='g', label='noise')
    plt.plot(T, XX, color='b', label='kalman')
    plt.show()

if __name__ == '__main__':
    # demo_kalman_xy()
    wave_kalman_xy()
