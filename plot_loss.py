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
    x = np.arange(length)*20
    y_0 = np.array(loss_list[0])
    y_1 = np.array(loss_list[1])
    y_2 = np.array(loss_list[2])
    y_3 = np.array(loss_list[3])
    
    plt.plot(x, y_0, 'r', label='rg_1')
    plt.plot(x, y_1, 'b', label='rg_2')
    plt.plot(x, y_2, 'g', label='rg_3')
    plt.plot(x, y_3, 'y', label='rg_4')

    plt.legend()

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()

def read():
    mypath = './log/'
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    loss_list = []
    for filename in files:
        if len(filename.split('_')) < 4:
            continue
        if filename.split('_')[3] != "rg":
            continue
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
