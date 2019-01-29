

import matplotlib.pyplot as plt
import math
import numpy as np
import os

###################################################################################

def read_plot_wave(rid):
    listT, listP = readWave(rid)
    GY, MY = lowPass(listP)
    GYY = highPass(GY)
    MYY = highPass(MY)
    print(MYY)

    plt.figure()
    plt.plot(listT, GYY, color = 'r', label = 'waveG')
    plt.plot(listT, MYY, color = 'g', label='waveM')
    plt.legend()
    plt.show()

def readWave(rid):
    file = os.path.join(os.getcwd(), 'files\\wave2.txt')

    first = 0
    size = 0
    origin_sample_time = []
    origin_sample_pixel = []
    interpolate_time = []
    interpolate_pixel = []
    pre_hamming_pixel = []

    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            lst = line.split(',')
            if lst[0] == '666' and lst[1] == rid:
                first = int(lst[2])
                size = int(lst[3])
            elif  lst[0] == '111' and lst[1] == rid:
                for i in range(2, len(lst) - 1):
                    origin_sample_time.append(float(lst[i]))
            elif lst[0] == '222' and lst[1] == rid:
                for i in range(2, len(lst) - 1):
                    origin_sample_pixel.append(float(lst[i]))
            elif lst[0] == '333' and lst[1] == rid:
                for i in range(2, len(lst) - 1):
                    interpolate_time.append(float(lst[i]))
            elif lst[0] == '444' and lst[1] == rid:
                for i in range(2, len(lst) - 1):
                    interpolate_pixel.append(float(lst[i]))
            elif lst[0] == '555' and lst[1] == rid:
                for i in range(2, len(lst) - 1):
                    pre_hamming_pixel.append(float(lst[i]))

    print(len(origin_sample_time))
    print(len(origin_sample_pixel))
    print(len(interpolate_time))
    print(len(interpolate_pixel))
    print(len(pre_hamming_pixel))

    origin_sample_time_sort = []
    origin_sample_pixel_sort = []
    for i in range(len(origin_sample_time)):
        origin_sample_time_sort.append(origin_sample_time[(i + first) % size])
        origin_sample_pixel_sort.append(origin_sample_pixel[(i + first) % size])
    print(len(origin_sample_time_sort))
    print(len(origin_sample_pixel_sort))
    print(origin_sample_time_sort)
    print(origin_sample_pixel_sort)

    return origin_sample_time_sort, origin_sample_pixel_sort, interpolate_time, interpolate_pixel, pre_hamming_pixel

def read_plot_wave(rid):
    X, Y, Z, W, U = readWave(rid)

    plt.figure()
    plt.plot(X, Y, color = 'r', label = 'origin-sample')
    plt.scatter(X, Y, color = 'g')

    plt.plot(Z, W, color = 'b', label = 'interpolate-sample')
    plt.scatter(Z, W, color = 'y')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(Z, U, color = 'r', label = 'hamming-sample')
    plt.scatter(Z, U, color = 'g')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    read_plot_wave('31401')
