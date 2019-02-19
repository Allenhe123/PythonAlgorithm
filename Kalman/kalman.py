
##################################
# 估计体重
##################################

import matplotlib.pyplot as plt
import math
import numpy as np

z = np.array([158.0, 164.2, 160.3, 159.9, 162.1, 164.6, 169.6, 167.4, 166.4, 171.0, 171.2, 172.6])
x = np.arange(160, 172, 1)
er = z - x
# plt.figure()
# plt.scatter(er, np.zeros(12))   # 画散点图
# plt.show()
print(x)
print('mean:', np.mean(er)) # 均值
print('var:', np.var(er))   # numpy中方差var、协方差cov


# intial parameters
z = np.array([158.0, 164.2, 160.3, 159.9, 162.1, 164.6, 169.6, 167.4, 166.4, 171.0, 171.2, 172.6])
A = 1
B = 1
u_k = 1
Q = 0
H = 1
R = 4.5
# allocate space for arrays
sz = 12
xhat = np.zeros(sz)      # a posteri estimate of x
P = np.zeros(sz)         # a posteri error estimate
xhatminus = np.zeros(sz) # a priori estimate of x
Pminus = np.zeros(sz)    # a priori error estimate
K = np.zeros(sz)         # gain or blending factor

# intial guesses
xhat[0] = 160 #设为真实值
P[0] = 1.0

for k in range(1,sz):
    # time update
    xhatminus[k] = A * xhat[k-1] + B * u_k  # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
    Pminus[k] = A * P[k-1] * A + Q          # P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1

    # measurement update
    K[k] = Pminus[k] * H / (H * Pminus[k] * H + R )       #Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
    xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k]) #X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
    P[k] = (1 - K[k] * H) * Pminus[k]                         #P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1

plt.figure()
plt.plot(z,'k+',label='noisy measurements')     #测量值
plt.plot(xhat,'b-',label='a posteri estimate')  #过滤后的值
plt.plot(x,color='g',label='truth value')       #系统值
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('weight')
plt.show()
