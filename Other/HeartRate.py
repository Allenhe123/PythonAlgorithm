

import matplotlib.pyplot as plt
import math
import numpy as np
import os


def dedup_list(li):

    dedupli = []
    index = 0
    for temp in li:
        if li.index(temp) == index:
            dedupli.append(temp)
        index += 1
    return dedupli

def remove_list(li, rid):
    retlist = []
    for str in li:
        strlst = str.split(" ")
        if strlst[0] in rid:
            retlist.append(str)
    return  retlist

def read_plot(filename, rid):
    with open(filename, 'r') as f:
        lists = f.readlines()
        lists = remove_list(lists, rid)

        vals = []
        times = []

        index = 0
        for i in range(0, len(lists)):
            line = lists[i].rstrip('\n')
            str = line.split(" ")
            vals.append(float(str[1]))
            times.append(index)
            index += 1

        print(filename, ': ', len(vals))
        plt.title(filename)
        plt.scatter(times, vals, color = 'g')
        plt.plot(times, vals)
        plt.show()

###################################################################################

def readWave(rid):
    file = os.path.join(os.getcwd(), 'files\\wave.txt')
    result = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            lst = line.split(' ')
            if lst[1] == rid:
                result.append(line)
    print(len(result))

    strlst = result[0].split(' ')
    starttime = float(strlst[0])
    print(starttime)

    listT = []
    listP = []
    listT.append(starttime)
    listP.append(float(strlst[2]))
    for l in result:
        lst = l.split(' ')
        t = float(lst[0])
        if t - starttime >= 10:
            break
        listT.append(float(lst[0]))
        listP.append(float(lst[2]))
    print(len(listT), len(listP))
    return  listT, listP

def lowPass(Y):
    Gauss = [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125]
    Mid = [1, 1, 1, 1, 1, 1, 1]
    GY = []   # 高斯滤波
    MY = []   # 均值滤波

    for i in range(len(Y)):
        value = 0
        valueM = 0
        wei = 0
        weiM = 0
        for k in range(i-3, i+4, 1):
            if k < 0 or k > len(Y) - 1:
                continue
            else:
                value += Y[k] * Gauss[k + 3 - i]
                valueM += Y[k] * Mid[k + 3 - i]
                wei += Gauss[k + 3 - i]
                weiM += Mid[k + 3 - i]
        if wei > 0:
            value = value / wei
            GY.append(value)
            MY.append(valueM / weiM)
    return  GY, MY

def highPass(Y):
    print(len(Y))
    listHP = []
    for i in range(len(Y) - 1):
        listHP.append(2 * (Y[i + 1] - Y[i]))
    listHP.append(2 * Y[len(Y) - 1])
    print(len(listHP))
    return  listHP

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

def readWave2(rid):
    file = os.path.join(os.getcwd(), 'files\\wave.txt')
    result = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            lst = line.split(' ')
            if lst[1] == rid:
                result.append(line)
    print(len(result))

    strlst = result[0].split(' ')
    id = int(strlst[0])
    starttime = float(strlst[2])
    print(starttime)

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

    print(listT[len(listT) - 1])
    deltaT = listT[len(listT) - 1] - listT[0]
    mean /= cnt
    print(deltaT)
    print(listT[1] - listT[0])
    print(listP[0])
    print(mean)
    print(min(listP), max(listP))

    return listT, listP


def read_plot_wave2(rid):
    X, Y = readWave2(rid)
    GY, MY = lowPass(Y)

    plt.figure()
    # plt.xticks(np.linspace(X[0], X[len(X) - 1], 100))
    # plt.yticks(np.linspace(min(Y), max(Y), 50))
    plt.plot(X, GY, color = 'r', label = 'waveG')
    plt.scatter(X, GY, color = 'g')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # print('heart rate')
    #
    # paths = ['E:\\reference.txt', 'E:\\facemean.txt']
    #
    # RID = ['16']
    #
    # for file in paths:
    #     read_plot(file, RID)
    # read_plot_wave('31401')
    read_plot_wave2('31401')
