#!/usr/bin/env python

# author: Wenkai Jiang
# date: 22 / 05 / 2017
# last modified: 22 / 05 / 2017
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

    train_epoch = 1000
    train_minibatch = 20
    precision = '_g32w32_2l_1'
    TEST_INFILE = './exp_result/' + 'e' + str(train_epoch) + 'mb' + str(train_minibatch) + precision + '.npy'
    TEST_OUTFILE = './test_result/' + 'e' + str(train_epoch) + 'mb' + str(train_minibatch) + precision

    with open(TEST_OUTFILE, 'a') as log:
        log.write('###########################################\n')
        log.write('# TEST RESULT' + '\n')
        log.write('# Train epoch number: ' + str(train_epoch) +'\n')
        log.write('# Train minibatch size: ' + str(train_minibatch) + '\n')
        log.write('# Train precision type: ' + 'low precision' +'\n')
        log.write('###########################################\n')

    D = image.shape[0] * image.shape[1]  # dimenstion 784, image size [28 x 28]
    K = 10  # number of classes

    # general info
    # print(len(testing_data))
    # print(label)
    # print(image.shape)

    W_1, b_1, W_2, b_2 = np.load(TEST_INFILE)
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
        error_count += test(labels, images, W_1, b_1, W_2, b_2)
        if (start/minibatch_size) % 5 == 0:
            info = "iteration %d : error_rate %f" % ((start/minibatch_size), error_count / start)
            print info
            with open(TEST_OUTFILE, 'a') as log:
                log.write(info +'\n')

def test(Y, X, W_1, b_1, W_2, b_2):
    hidden_output = np.dot(X, W_1) + b_1
    hidden_X = np.tanh(hidden_output)
    scores = np.dot(hidden_X, W_2) + b_2
    # print scores

    # get unnormalized probabilities
    scores = scores - scores.max(axis=1, keepdims=True)
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
    HK = 500
    K = 10  # number of classes

    # full precision parameters
    rng = np.random.RandomState(1234)

    W_1 = np.asarray(
        rng.uniform(
            low = -np.sqrt(6. / (D+HK)),
            high = np.sqrt(6. / (D+HK)),
            size = (D, HK)
        )
    )   # [784 x 500]
    b_1 = np.zeros((1,HK)) # [1 x 500]

    W_2 = 0.01 * np.random.randn(HK,K) # [500 x 10]
    # W_2 = np.zeros((HK,K)) # [500 x 10]
    b_2 = np.zeros((1,K)) # [1 x 10]

    # low precision parameters
    W_low_1 = W_1
    b_low_1 = b_1

    W_low_2 = W_2
    b_low_2 = b_2

    # set a upbound for index of dataset
    start_max = len(training_data)

    # some hyperparameters
    epoch = 10
    minibatch = 150

    # for debug, set a upbound for index of dataset
    # start_max = 1
    # epoch = 1
    # minibatch = 1

    learning_rate = 1e-3

    # initialize mt and vt for Adam algorithm
    m_wt_1 = np.zeros((D,HK))
    v_wt_1 = m_wt_1

    m_bt_1 = np.zeros((1,HK))
    v_bt_1 = m_bt_1

    m_wt_2 = np.zeros((HK,K))
    v_wt_2 = m_wt_2

    m_bt_2 = np.zeros((1,K))
    v_bt_2 = m_bt_2

    # precision = '_g5w5_2l_1'
    precision = '_sgd'
    PARAM_OUTFILE = './exp_result/' + 'e' + str(epoch) + 'mb' + str(minibatch) + precision
    LOG_OUTFILE = './log/'+'e' + str(epoch) + 'mb' + str(minibatch) + precision
    log_step = 20

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
            # if len(training_data)-start+1 < minibatch_size:
            #     minibatch_size = len(training_data)-start+1

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

            # ################################################################################
            # # g5w5
            # # Below code, where lies the only difference between full precision and low one
            # # use low precision parameters to get full precision gradients;
            # # simulate on the worker side
            # dW_1, db_1, dW_2, db_2, loss_temp= train(labels, images, W_low_1, b_low_1, W_low_2, b_low_2, minibatch_size)
            # loss += loss_temp  # accumulate loss
            #
            # # reduce precision of gradients
            # dW_low_1 = to_low_pcs(dW_1, 5)
            # db_low_1 = to_low_pcs(db_1, 5)
            #
            # dW_low_2 = to_low_pcs(dW_2, 5)
            # db_low_2 = to_low_pcs(db_2, 5)
            #
            # # update parameter with low precision gradients
            # W_1, m_wt_1, v_wt_1 = adam(W_1, dW_low_1, learning_rate, m_wt_1, v_wt_1, (start_max / minibatch_size)*i+(start/minibatch_size))
            # b_1, m_bt_1, v_bt_1 = adam(b_1, db_low_1, learning_rate, m_bt_1, v_bt_1, (start_max / minibatch_size)*i+(start/minibatch_size))
            #
            # W_2, m_wt_2, v_wt_2 = adam(W_2, dW_low_2, learning_rate, m_wt_2, v_wt_2, (start_max / minibatch_size)*i+(start/minibatch_size))
            # b_2, m_bt_2, v_bt_2 = adam(b_2, db_low_2, learning_rate, m_bt_2, v_bt_2, (start_max / minibatch_size)*i+(start/minibatch_size))
            #
            # # Change W and b to low precision; simulate on the server side
            # W_low_1 = to_low_pcs(W_1, 5)
            # b_low_1 = to_low_pcs(b_1, 5)
            #
            # W_low_2 = to_low_pcs(W_2, 5)
            # b_low_2 = to_low_pcs(b_2, 5)
            # ###############################################################################

            # ################################################################################
            # # g32w5
            # # Below code, where lies the only difference between full precision and low one
            # # use low precision parameters to get full precision gradients;
            # # simulate on the worker side
            # dW_1, db_1, dW_2, db_2, loss_temp= train(labels, images, W_low_1, b_low_1, W_low_2, b_low_2, minibatch_size)
            # loss += loss_temp  # accumulate loss
            #
            # # update parameter with low precision gradients
            # W_1, m_wt_1, v_wt_1 = adam(W_1, dW_1, learning_rate, m_wt_1, v_wt_1, (start_max / minibatch_size)*i+(start/minibatch_size))
            # b_1, m_bt_1, v_bt_1 = adam(b_1, db_1, learning_rate, m_bt_1, v_bt_1, (start_max / minibatch_size)*i+(start/minibatch_size))
            #
            # W_2, m_wt_2, v_wt_2 = adam(W_2, dW_2, learning_rate, m_wt_2, v_wt_2, (start_max / minibatch_size)*i+(start/minibatch_size))
            # b_2, m_bt_2, v_bt_2 = adam(b_2, db_2, learning_rate, m_bt_2, v_bt_2, (start_max / minibatch_size)*i+(start/minibatch_size))
            #
            # # Change W and b to low precision; simulate on the server side
            # W_low_1 = to_low_pcs(W_1, 5)
            # b_low_1 = to_low_pcs(b_1, 5)
            #
            # W_low_2 = to_low_pcs(W_2, 5)
            # b_low_2 = to_low_pcs(b_2, 5)
            # ###############################################################################
            #
            ################################################################################
            # g32w32
            # Below code, where lies the only difference between full precision and low one
            # use low precision parameters to get full precision gradients;
            # simulate on the worker side
            dW_1, db_1, dW_2, db_2, loss_temp= train(labels, images, W_1, b_1, W_2, b_2, minibatch_size)
            loss += loss_temp  # accumulate loss

            # SGD
            # W_1 += -learning_rate * dW_1
            # b_1 += -learning_rate * db_1
            # W_2 += -learning_rate * dW_2
            # b_2 += -learning_rate * db_2

            # update parameter with low precision gradients
            W_1, m_wt_1, v_wt_1 = adam(W_1, dW_1, learning_rate, m_wt_1, v_wt_1, (start_max / minibatch_size)*i+(start/minibatch_size))
            b_1, m_bt_1, v_bt_1 = adam(b_1, db_1, learning_rate, m_bt_1, v_bt_1, (start_max / minibatch_size)*i+(start/minibatch_size))

            W_2, m_wt_2, v_wt_2 = adam(W_2, dW_2, learning_rate, m_wt_2, v_wt_2, (start_max / minibatch_size)*i+(start/minibatch_size))
            b_2, m_bt_2, v_bt_2 = adam(b_2, db_2, learning_rate, m_bt_2, v_bt_2, (start_max / minibatch_size)*i+(start/minibatch_size))

            W_low_1 = W_1
            b_low_1 = b_1
            W_low_2 = W_2
            b_low_2 = b_2
            ###############################################################################

            if (start/minibatch_size) % log_step == 0:
                info = "epoch %2d : iteration %4d : loss %f" % (i+1, (start/minibatch_size), loss / log_step)
                print info
                with open(LOG_OUTFILE, 'a') as log:
                    log.write(info + '\n')
                loss = 0
                np.save(PARAM_OUTFILE, (W_low_1, b_low_1, W_low_2, b_low_2))   # x,y,z equal sized 1D arrays

def train(Y, X, W_1, b_1, W_2, b_2, batch_size):
    # hyperparameters
    reg = 2e-4 # regularization strength

    hidden_output = np.dot(X, W_1) + b_1
    hidden_X = np.tanh(hidden_output)
    scores = np.dot(hidden_X, W_2) + b_2 # [batch_size x class number]
    # print scores

    # get unnormalized probabilities
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    # normalize them for each example
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # compute the loss: average cross-entropy loss and regularization
    corect_logprobs = -np.log(probs[range(batch_size), Y]) # [N x 1]
    data_loss = np.sum(corect_logprobs)/batch_size
    reg_loss = 0.5 * reg * ((W_1 ** 2).sum() + (W_2 ** 2).sum())
    loss = data_loss + reg_loss

    # compute the gradient on scores
    dscores = probs
    dscores[range(batch_size), Y] -= 1
    dscores /= batch_size

    # backpropate the gradient to the parameters (W,b)
    dW_2 = np.dot(hidden_X.T, dscores)
    db_2 = np.sum(dscores, axis=0, keepdims=True)
    dW_2 += reg * W_2 # regularization gradient

    dhidden_X = np.dot(dscores, W_2.T)
    dhidden_output = np.multiply(dhidden_X, (1 - hidden_X ** 2))

    dW_1 = np.dot(X.T, dhidden_output)
    db_1 = np.sum(dhidden_output, axis=0, keepdims=True)
    dW_1 += reg * W_1

    return (dW_1, db_1, dW_2, db_2, loss)

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
