import matplotlib.pyplot as plt
import math
import numpy as np


class BpmSample:
    def __init__(self):
        self.idx = []
        self.mean = []
        self.mid = []
        self.min = []
        self.max = []
        self.stat = []

    def load(self, filename):
        with open(filename, 'r') as f:
            lists = f.readlines()

            idx = 0
            for i in range(0, len(lists)):
                line = lists[i].rstrip('\n')
                str = line.split(",")

                tmp = []
                for j in range(0, len(str)):
                    key = str[j]
                    if key == '':
                        continue
                    tmp.append(int(key))

                tmp.sort()

                minv = tmp[0]
                maxv = tmp[int(len(tmp) - 1)]
                midv = tmp[int(len(tmp) / 2)]
                self.min.append(minv)
                self.max.append(maxv)
                self.mid.append(midv)

                sum1 = 0
                sum2 = 0
                cnt = 0
                for v in tmp:
                    sum1 += v
                    if (v == minv or v == maxv):
                        continue
                    else:
                        sum2 += v
                        cnt += 1

                self.mean.append(sum1 / len(tmp))
                self.stat.append(sum2 / cnt)

                self.idx.append(idx)
                idx += 1

                if idx > 100:
                    break

    # def process(self):


    def draw(self):
        plt.figure()
        # plt.plot(self.idx, self.mean, color='r', label='mean')
        # plt.plot(self.idx, self.mid, color='g', label='mid')
        # plt.plot(self.idx, self.min, color='b', label='min')
        # plt.plot(self.idx, self.max, color='gold', label='max')
        plt.plot(self.idx, self.stat, color='g', label='stat')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    bpm = BpmSample()
    bpm.load('files/wave.txt')
    bpm.draw()
