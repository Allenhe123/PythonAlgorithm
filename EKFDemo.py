
##################################
# 估计轨迹
# 基本的卡尔曼滤波只能对线性的轨迹进行滤波！！！
# 非线性的轨迹需要用扩展卡尔曼滤波（EKF）
##################################

import matplotlib.pyplot as plt
import math
import random
import numpy as np

t = 10        # 仿真时间
Ts = 0.1      # 采样周期
len = t / Ts  # 仿真步数

############################################################################
X1 = np.zeros((int(len), 6))
X1[0,:] = np.array([0, 0, 1, 1, 0, 0]) # 状态模拟的初值(x, y, vx, vy, ax, ay)
Z1 = np.zeros((int(len), 2))

X2 = np.zeros((int(len), 6))
X2[0,:] = np.array([1, 1, 2, 2, 0, 0]) # 状态模拟的初值(x, y, vx, vy, ax, ay)
Z2 = np.zeros((int(len), 2))

for i in range(1, int(len)):
    x0 = X1[i - 1, 0]
    y0 = X1[i - 1, 1]
    vx0 = X1[i - 1, 2]
    vy0 = X1[i - 1, 3]
    ax0 = X1[i - 1, 4]
    ay0 = X1[i - 1, 5]
    x1 = x0 + vx0 * Ts + 0.5 * ax0 * Ts**2
    y1 = y0 + vy0 * Ts + 0.5 * ay0 * Ts**2
    vx1 = vx0 + ax0 * Ts
    vy1 = vy0 + ay0 * Ts
    ax1 = ax0 + (random.random() - 0.5) * 2
    ay1 = ay0 + (random.random() - 0.5) * 2
    X1[i,:] = [x1, y1, vx1, vy1, ax1, ay1]

for k in range(0, int(len)):
    x = X1[k, 0] +  1 * (random.random() - 0.5)
    y = X1[k, 1] +  1 * (random.random() - 0.5)
    Z1[k:] = [x, y]
#######################################################
for i in range(1, int(len)):
    x0 = X1[i - 1, 0]
    y0 = X1[i - 1, 1]
    vx0 = X1[i - 1, 2]
    vy0 = X1[i - 1, 3]
    ax0 = X1[i - 1, 4]
    ay0 = X1[i - 1, 5]
    x1 = x0 + vx0 * Ts + 0.5 * ax0 * Ts**2
    y1 = y0 + vy0 * Ts + 0.5 * ay0 * Ts**2
    vx1 = vx0 + ax0 * Ts
    vy1 = vy0 + ay0 * Ts
    ax1 = ax0 + (random.random() - 0.5) * 2
    ay1 = ay0 + (random.random() - 0.5) * 2
    X2[i,:] = [x1, y1, vx1, vy1, ax1, ay1]

for k in range(0, int(len)):
    x = X1[k, 0] +  3 * (random.random() )
    y = X1[k, 1] +  3 * (random.random() )
    Z2[k:] = [x, y]

##################################################################################

observed_x1 = Z1[:,0]
observed_y1 = Z1[:,1]

observed_x2 = Z2[:,0]
observed_y2 = Z2[:,1]

####################################################################################

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

def demo_kalman_xy(observed_x, observed_y):
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

    return kalman_x, kalman_y

    plt.plot(X1[:, 0], X1[:, 1])
    plt.plot(observed_x, observed_y, 'bo')
    plt.plot(kalman_x, kalman_y, color = 'r')
    plt.show()

##################################################################################################
# print(X1)
# print(Z1)

# demo_kalman_xy(observed_x1, observed_y1)
# print(X1.shape)
# print(Z1.shape)

test_x1 = []
test_y1 = []

test_x2 = []
test_y2 = []

print(Z1.shape[0] / 2)

for i in range(math.ceil(Z1.shape[0] / 2)):
    test_x1.append(Z1[i, 0])
    test_y1.append(Z1[i, 1])

for i in range(math.ceil(Z2.shape[0] / 2 + 1)):
    test_x2.append(Z2[i, 0])
    test_y2.append(Z2[i, 1])

test_x1.append(Z2[math.ceil(Z2.shape[0] / 2 + 1) - 1, 0])
test_y1.append(Z2[math.ceil(Z2.shape[0] / 2 + 1) - 1, 1])

print('testX1: ', test_x1)
print('testY1: ', test_y1)
print('lastCoordX1_origin: ', Z1[math.ceil(Z1.shape[0] / 2 + 1) - 1, 0])
print('lastCoordX1_origin: ', Z1[math.ceil(Z1.shape[0] / 2 + 1) - 1, 1])

print('lastCoordX1: ', test_x1[-1])
print('lastCoordY1: ', test_y1[-1])

kalmanX1, kalmanY1 =  demo_kalman_xy(test_x1, test_y1)
print('kalmanX1: ', kalmanX1[-1])
print('kalmanY1: ', kalmanY1[-1])

print('')
print('testX2: ', test_x2)
print('testY2: ', test_y2)
print('lastCoordX2: ', test_x2[-1])
print('lastCoordY2: ', test_y2[-1])

kalmanX2, kalmanY2 = demo_kalman_xy(test_x2, test_y2)
print('kalmanX2: ', kalmanX2[-1])
print('kalmanY2: ', kalmanY2[-1])

##### distance
d1 = math.pow(Z1[math.ceil(Z1.shape[0] / 2 + 1) - 1, 0] - kalmanX1[-1], 2) \
     + math.pow(Z1[math.ceil(Z1.shape[0] / 2 + 1) - 1, 1] - kalmanY1[-1], 2)

d2 = math.pow(Z1[math.ceil(Z1.shape[0] / 2 + 1) - 1, 0] - kalmanX2[-1], 2) \
     + math.pow(Z1[math.ceil(Z1.shape[0] / 2 + 1) - 1, 1] - kalmanY2[-1], 2)

d3 = math.pow(test_x2[-1] - kalmanX1[-1], 2) + math.pow(test_y2[-1] - kalmanY1[-1], 2)

d4 = math.pow(test_x2[-1] - kalmanX2[-1], 2) + math.pow(test_y2[-1] - kalmanY2[-1], 2)

print('轨迹1取前49个点，加上轨迹2的第50个点。轨迹2取前50 个点。然后对分别对2个轨迹的第50个点进行卡尔曼滤波。。。')
print('轨迹1原来的第50个点与轨迹1中添加的轨迹2的第50个点滤波后坐标的距离：', d1)
print('轨迹1原来的第50个点与轨迹2的第50个点滤波后坐标的距离：', d2)
print('轨迹2的第50个点与轨迹1中添加的轨迹2的第50个点滤波后坐标的距离：', d3)
print('轨迹2的第50个点与轨迹2的第50个点滤波后坐标的距离：', d4)
