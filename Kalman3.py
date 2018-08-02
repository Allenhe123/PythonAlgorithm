
##################################
# 估计轨迹
# 基本的卡尔曼滤波只能对线性的轨迹进行滤波！！！
# 非线性的轨迹需要用扩展卡尔曼滤波（EKF）
##################################

import matplotlib.pyplot as plt
import math
import random
import numpy as np

PI = np.pi
delta = PI / 10
a = []
b = []
c = []
for i in range(0, 50):
    a.append(i)
    b.append(math.sin(delta * i));
    err = random.random() - 0.5
    c.append(math.sin(delta * i) + err)

za = np.array(a)
zb = np.array(b)
zx = np.array(c)

# intial parameters
A = 1
B = 1
u_k = 1
Q = 0.018
H = 1
R = 0.542
# allocate space for arrays
sz = 50
xhat = np.zeros(sz)      # a posteri estimate of x
Px = np.zeros(sz)         # a posteri error estimate
Py = np.zeros(sz)
xhatminus = np.zeros(sz) # a priori estimate of x
PminusX = np.zeros(sz)    # a priori error estimate
Kx = np.zeros(sz)         # gain or blending factor


# intial guesses
xhat[0] = 1 #设为真实值
Px[0] = 1.0
Py[0] = 1.0

for k in range(1, sz):
    # time update
    xhatminus[k] = A * xhat[k-1] + B * u_k  # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0

    PminusX[k] = A * Px[k-1] * A + Q          # P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1

    # measurement update
    Kx[k] = PminusX[k] * H / (H * PminusX[k] * H + R )       #Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1

    xhat[k] = xhatminus[k] + Kx[k] * (zx[k] - xhatminus[k]) #X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1

    Px[k] = (1 - Kx[k]) * PminusX[k]                         #P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1

plt.figure()
plt.plot(za, zx, label='noisy measurements')                #测量值
plt.plot(za, xhat, color = 'g', label='kalman estimate')    #过滤后的值
plt.plot(za, zb, color='r',label='truth value')             #系统值
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('weight')
plt.show()
