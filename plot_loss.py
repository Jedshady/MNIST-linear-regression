#!usr/bin/python

# author: Wenkai Jiang
# date: 22 / 05 / 2017
# last modified: 22 / 05 / 2017
# location: NUS

from os import listdir
from os.path import isfile, join

import numpy as np
import matplotlib.pyplot as plt

def main():
    loss_list = read()
    length = len(loss_list[0])

    # l = ['2', '0.7', '0.3']
    x = np.arange(length)*50
    y_0 = np.array(loss_list[0])
    y_1 = np.array(loss_list[1])
    y_2 = np.array(loss_list[2])
    y_3 = np.array(loss_list[3])
    y_4 = np.array(loss_list[4])
    y_5 = np.array(loss_list[5])
    plt.plot(x, y_3, 'r', label='1_worker')
    plt.plot(x, y_0, 'b', label='2_workers')
    plt.plot(x, y_1, 'g', label='3_workers')
    plt.plot(x, y_2, 'y', label='4_workers')
    plt.plot(x, y_4, 'm', label='5_workers')
    plt.plot(x, y_5, 'k', label='8_workers')
    plt.ylim([0, 2.5])
    plt.legend()

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()

def read():
    mypath = './log/test2/temp/'
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    loss_list = []
    for filename in files:
        # if len(filename.split('_')) < 4:
        #     continue
        # if filename.split('_')[3] != "rg":
        #     continue
        with open(join(mypath,filename), 'r') as f:
            contents = f.readlines()
        print filename
        loss = [line.split(':')[2].split()[1] for line in contents if line[0][0] is not '#']
        # for line in contents:
        #     if line[0][0] is not '#':
        #         print line.split(':')[2].split()[1]
        loss_list.append(loss)

    return loss_list

if __name__ == '__main__':
    main()
