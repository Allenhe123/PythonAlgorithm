import os
import numpy as np
import math
import scipy as sp
import scipy.stats as gamma
import matplotlib.pyplot as plt

class TSV:
    data_dir = ''
    x = []
    y = []

    def genData(self):
        sp.random.seed(3)  # to reproduce the data later on
        self.x = sp.arange(1, 31 * 24)
        self.y = sp.array(200 * (sp.sin(2 * sp.pi * self.x / (7 * 24))), dtype=float)
        # dice = gamma.rv_discrete(values=(self.x, self.y))
        # self.y += dice.rvs(15, loc=0, scale=100, size=len(self.x))
        # self.y += gamma.rv_discrete.rvs(15, loc=0, scale=100, size=len(self.x))
        self.y += 2 * sp.exp(self.x / 100.0)
        self.y = sp.ma.array(self.y, mask=[self.y < 0])
        print(sum(self.y), sum(self.y < 0))

    def draw(self):
        plt.scatter(self.x, self.y)   # 绘制散点图  https://blog.csdn.net/qi_1221/article/details/73903131
        plt.title("Web traffic over the last month")
        plt.xlabel("Time")
        plt.ylabel("Hits/hour")
        # set the current tick locations and labels of the x-axis. 人为设置坐标轴的刻度显示的值。
        plt.xticks([w * 7 * 24 for w in [0, 1, 2, 3, 4]], ['week %i' % (w + 1) for w in [
            0, 1, 2, 3, 4]])
        plt.autoscale(tight=True)
        plt.grid()
        plt.show()
        plt.savefig(os.path.join("..", "1400_01_01.png"))

    def save(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "files")
        sp.savetxt(os.path.join(self.data_dir, "web_traffic.tsv"), list(zip(self.x, self.y)), delimiter="\t", fmt="%s")

'''
        os.path.realpath(__file__): 获得的是该方法所在的脚本的路径
        os.getcwd(): 获得的是当前执行脚本的所在路径，无论从哪里调用的该方法。
        (1).当"print os.path.dirname(__file__)"所在脚本是以完整路径被运行的， 那么将输出该脚本所在的完整路径，比如：
             python d:/pythonSrc/test/test.py
             那么将输出 d:/pythonSrc/test
        (2).当"print os.path.dirname(__file__)"所在脚本是以相对路径被运行的， 那么将输出空目录，比如：
             python test.py
             那么将输出空字符串

        zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回一个对象。
        在 Python 3.x 中为了减少内存，zip() 返回的是一个对象。如需展示列表，需手动 list() 转换成元组的列表。
        list() 方法用于将元组转换为列表。
'''

if __name__ == '__main__':
    tsv = TSV()
    tsv.genData()
    tsv.draw()
    tsv.save()
