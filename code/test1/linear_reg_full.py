#!/usr/bin/env python

# author: Wenkai Jiang
# date: 21 / 05 / 2017
# last modified: 21 / 05 / 2017
# location: NUS

"""
Acknowlegement:
Functions 'read()' and 'show()' are highly inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

import os
import random
import struct
import numpy as np
import math

def main():
    # comment to leave only one line
    trainProc() # for training
    # testProc() # for testing

def testProc():
    # read in test data
    testing_data = list(read('testing'))
    label,image = testing_data[0]

    train_epoch = 10
    train_minibatch = 150
    precision = '_full'
    TEST_INFILE = './exp_result/' + 'e' + str(train_epoch) + 'mb' + str(train_minibatch) + precision + '.npy'
    TEST_OUTFILE = './test_result/' + 'e' + str(train_epoch) + 'mb' + str(train_minibatch) + precision

    with open(TEST_OUTFILE, 'a') as log:
        log.write('###########################################\n')
        log.write('# TEST RESULT' + '\n')
        log.write('# Train epoch number: ' + str(train_epoch) +'\n')
        log.write('# Train minibatch size: ' + str(train_minibatch) + '\n')
        log.write('# Train precision type: ' + 'full precision' +'\n')
        log.write('###########################################\n')

    D = image.shape[0] * image.shape[1]  # dimenstion 784, image size [28 x 28]
    K = 10  # number of classes

    # general info
    # print(len(testing_data))
    # print(label)
    # print(image.shape)

    W,b = np.load(TEST_INFILE)
    error_count = 0.0
    test_minibatch = 100
    start = 0

    while start < len(testing_data):
        minibatch_size = test_minibatch

        images = np.zeros((minibatch_size,D)) # data matrix (each row = single example)
        labels = np.zeros(minibatch_size, dtype='uint8') # class labels
        for i in xrange(minibatch_size):
            label,image = testing_data[start]
            X = image.flatten().reshape((1,D)) # [1 x 784] int
            newImage = np.zeros((1,D), dtype=np.float32) # [1 x 784] float32
            for pos in xrange(D):
                newImage[0,pos] = X[0,pos]/255.0
            labels[i] = label
            images[i] = newImage
            start += 1
        error_count += test(labels, images, W, b)
        if (start/minibatch_size) % 5 == 0:
            info = "iteration %d : error_rate %f" % ((start/minibatch_size), error_count / start)
            print info
            with open(TEST_OUTFILE, 'a') as log:
                log.write(info +'\n')

def test(Y, X, W, b):
    scores = np.dot(X, W) + b
    # print scores

    # get unnormalized probabilities
    exp_scores = np.exp(scores)
    # normalize them for each example
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    predict = np.argmax(probs, axis=1)
    res = predict - Y
    error = np.count_nonzero(res)
    return error

def trainProc():
    # Read in data
    training_data = list(read('training'))
    label,image = training_data[0]

    # general info for debugging
    # print(len(training_data))
    # print(label)
    # print(image.shape)

    # dimenstion of data
    D = image.shape[0] * image.shape[1]  # dimenstion 784, image size [28 x 28]
    K = 10  # number of classes

    # full precision parameters
    W = 0.01 * np.random.randn(D,K) # [784 x 10]
    b = np.zeros((1,K)) # [1 x 10]

    # low precision parameters
    # W_low = W
    # b_low = b

    # for debug, set a upbound for index of dataset
    start_max = len(training_data)
    # start_max = 1

    # some hyperparameters
    epoch = 10
    minibatch = 150
    learning_rate = 1e-3

    # initialize mt and vt for Adam algorithm
    m_wt = np.zeros((D,K))
    v_wt = m_wt

    m_bt = np.zeros((1,K))
    v_bt = m_bt

    precision = '_full'
    PARAM_OUTFILE = './exp_result/' + 'e' + str(epoch) + 'mb' + str(minibatch) + precision
    LOG_OUTFILE = './log/'+'e' + str(epoch) + 'mb' + str(minibatch) + precision

    with open(LOG_OUTFILE, 'a') as log:
        log.write('###########################################\n')
        log.write('#epoch number: ' + str(epoch) +'\n')
        log.write('#minibatch size: ' + str(minibatch) + '\n')
        log.write('#precision type: ' + 'full precision' +'\n')
        log.write('###########################################\n')

    loss = 0

    for i in xrange(epoch):
        random.shuffle(training_data)   # at the beginning of every epoch, shuffle index
        minibatch_size = minibatch

        start = 0

        # iterate on all data at size of minibatch size
        while start < start_max:
            # in case number of data can not divided by minibatch size
            if len(training_data)-start+1 < minibatch_size:
                minibatch_size = len(training_data)-start+1

            # images -> X, labels -> Y
            images = np.zeros((minibatch_size,D)) # data matrix (each row = single example)
            labels = np.zeros(minibatch_size, dtype='uint8') # class labels
            for j in xrange(minibatch_size):
                label,image = training_data[start]
                X = image.flatten().reshape((1,D)) # [1 x 784] int
                # rescale image to [0, 1]
                newX = np.zeros((1,D), dtype=np.float32) # [1 x 784] float32
                for pos in xrange(D):
                    newX[0,pos] = X[0,pos]/255.0
                labels[j] = label
                images[j] = newX
                start += 1

            ################################################################################
            # From line 169 to line 181, where lies the only difference between full precision and low one
            # use low precision parameters to get full precision gradients;
            # simulate on the worker side
            dW, db, loss_temp= train(labels, images, W, b, minibatch_size)
            loss += loss_temp  # accumulate loss

            # perform a parameter update;
            # simulate on the server side
            W, m_wt, v_wt = adam(W, dW, learning_rate, m_wt, v_wt, (start_max / minibatch_size)*i+(start/minibatch_size))
            b, m_bt, v_bt = adam(b, db, learning_rate, m_bt, v_bt, (start_max / minibatch_size)*i+(start/minibatch_size))

            # Change W and b to low precision; simulate on the server side
            # W_low = to_low_pcs(W, 5)
            # b_low = to_low_pcs(b, 5)

            if (start/minibatch_size) % 20 == 0:
                info = "epoch %2d : iteration %4d : loss %f" % (i+1, (start/minibatch_size), loss / 20)
                print info
                with open(LOG_OUTFILE, 'a') as log:
                    log.write(info + '\n')
                loss = 0
                np.save(PARAM_OUTFILE, (W,b))   # x,y,z equal sized 1D arrays

def train(Y, X, W, b, batch_size):
    # hyperparameters
    reg = 1e-3 # regularization strength

    scores = np.dot(X, W) + b
    # print scores

    # get unnormalized probabilities
    exp_scores = np.exp(scores)
    # normalize them for each example
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # compute the loss: average cross-entropy loss and regularization
    corect_logprobs = -np.log(probs[range(batch_size), Y]) # [N x 1]
    data_loss = np.sum(corect_logprobs)/batch_size
    reg_loss = 0.5*reg*np.sum(W * W)
    loss = data_loss + reg_loss

    # compute the gradient on scores
    dscores = probs
    dscores[range(batch_size), Y] -= 1
    dscores /= batch_size

    # backpropate the gradient to the parameters (W,b)
    dW = np.dot(X.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)
    dW += reg * W # regularization gradient

    return (dW, db, loss)


def adam(parameter, grad, step_size, m_t, v_t, t):
    beta_1 = 0.9
    beta_2 = 0.999
    beta_1_t = np.power(beta_1, t)
    beta_2_t = np.power(beta_2, t)
    epsilon = 1e-8

    mt_biased = beta_1*m_t + (1-beta_1)*grad
    vt_biased = beta_2*v_t + (1-beta_2)*np.power(grad,2)

    mt_unbiased = mt_biased / (1 - beta_1_t)
    vt_unbiased = vt_biased / (1 - beta_2_t)

    param = parameter - step_size * mt_unbiased / (np.sqrt(vt_unbiased) + epsilon)
    return (param, mt_biased, vt_biased)


def to_low_pcs(parameter, bit):
    param = np.ndarray.copy(parameter)
    s = np.absolute(param).max()

    n_1 = np.floor(np.log(4*s/3)/np.log(2))
    n_2 = n_1 + 1 - np.power(2, bit-2)
    P = np.linspace(n_2, n_1, (n_1-n_2+1))
    P = np.power(2, P)
    P = np.append([0], P)

    # for i in xrange(len(P.tolist())):
        # print P[i]

    for x in np.nditer(param, op_flags=['readwrite']):
        sign = 1
        if x < 0:
            sign = -1

        abs_x = np.absolute(x)

        for i in xrange(len(P)-1):
            left = (P[i] + P[i+1]) / 2
            right = P[i+1] * 3 / 2
            if left <= abs_x and abs_x < right:
                x[...] = sign * P[i+1]
                break
        if i == 7 and x == abs_x * sign:
            x[...] = 0

    return param

def ascii_show(image):
    for y in image:
         row = ""
         for x in y:
             row += '{0: <4}'.format(x)
         print row

def read(dataset = "training", path = "./ds/"):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in xrange(len(lbl)):
        yield get_img(i)

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

if __name__ == '__main__':
    main()
