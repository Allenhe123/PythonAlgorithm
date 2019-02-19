import matplotlib.pyplot as plt
import math
import numpy as np

A1 = 0.4
A2 = 0.05
w1 = 1.57
w2 = 9.42
o2 = 0.956

X = np.arange(0, 15, 0.1)
Y = []

for i in X:
    Y.append(A1 * math.sin(i * w1) + A2 * math.sin(w2 * i + o2))


Gauss = [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125]
Mid = [1, 1, 1, 1, 1, 1, 1]
GY = []   # 高斯滤波
MY = []   # 均值滤波
MMY = []  # 中值滤波

for i in range(len(Y)):
    value = 0
    valueM = 0
    wei = 0
    mmy = []
    for k in range(i-3, i+4, 1):
        if k < 0 or k > len(Y) - 1:
            continue
        else:
            value += Y[k] * Gauss[k + 3 - i]
            wei += Gauss[k + 3 - i]

            valueM += Y[k] * Mid[k + 3 - i]

            mmy.append(Y[k])

    if wei > 0:
        value = value / wei
        GY.append(value)
        MY.append(valueM / 7)
    mmy.sort()
    MMY.append(mmy[3])

plt.figure()
plt.plot(X, Y, color = 'r', label = 'origin wave')
plt.plot(X, GY, color = 'g', label = 'Gauss filter')
plt.plot(X, MY, color = 'b', label = 'mean filter')
plt.plot(X, MMY, color = 'gold', label = 'mid filter')
plt.legend()
plt.show()