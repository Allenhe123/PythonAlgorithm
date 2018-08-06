

import matplotlib.pyplot as plt
import math
import numpy as np


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


if __name__ == '__main__':
    print('heart rate')

    paths = ['E:\\reference.txt', 'E:\\facemean.txt']

    RID = ['16']

    for file in paths:
        read_plot(file, RID)

    # a = []
    # b = []
    # for i in range(0, 50):
    #     if i % 2 == 0:
    #         a.append(140)
    #     else:
    #         a.append(130)
    #
    #     b.append(i)
    #
    # print(a)
    # print(b)
    # plt.plot(b, a)
    # plt.show()

    # delta = 3.1415926 / 10.0;
    # x = np.arange(0, 2 * np.pi, 0.01)
    # y = np.sin(x)
    #
    # x = []
    # y = []
    # cnt = 0
    # delta  = np.pi / 10;
    # for i in range(0, 100):
    #      x.append(i)
    #      y.append(math.fabs(np.sin(i * delta)))
    #
    # plt.plot(x, y)
    # plt.show()