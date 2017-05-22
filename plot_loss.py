#!usr/bin/python

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
    plt.plot(x, y_0, 'r', label='g32w5')
    plt.plot(x, y_1, 'b', label='g5w5')
    plt.plot(x, y_2, 'g', label='g32w32')
    plt.legend()

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()

def read():
    mypath = './log/'
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    loss_list = []
    for filename in files:
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
