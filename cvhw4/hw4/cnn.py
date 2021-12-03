import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main
import random

def get_mini_batch(im_train, label_train, batch_size):
    # TO DO

    m, n = len(im_train), len(im_train[0])

    index = np.arange(n)
    random.shuffle(index)
    batch_num = math.ceil(n // batch_size)

    mini_batch_x = []
    mini_batch_y = []

    for i in range(batch_num):
        cur_index = index[i*batch_size: min((i+1)*batch_size, n+1)]

        mini_batch_x.append(im_train[:, cur_index])

        per_batch = []
        for i in range(len(cur_index)):
            ten_col = [0] * 10
            cur_one_index = label_train[0][cur_index[i]]
            ten_col[cur_one_index] = 1
            per_batch.append(ten_col)

            # if cur_one_index == 0:
            #     per_batch.append([1] + [0]*9)
            # else:
            #     per_batch.append( [0] *(cur_one_index-1) + [1] + [0]*(10-cur_one_index))

        mini_batch_y.append(np.array(per_batch).T)

    return mini_batch_x, mini_batch_y


def fc(x, w, b):
    # TO DO

    y = np.matmul(w,x) + b

    return y


def fc_backward(dl_dy, x, w, b, y):
    # TO DO

    dl_dx = np.matmul(dl_dy, w)
    dl_dw = np.zeros(w.shape)

    for i in range(len(w)):
        for j in range(len(x)):
            dl_dw[i][j] = dl_dy[i] * x[j]

    # dl_dw = dl_dw.reshape(-1)
    dl_db = dl_dy

    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):
    # TO DO
    l = np.sum((y_tilde - y) ** 2)
    # derivative
    dl_dy = 2 * (y_tilde - y)

    return l, dl_dy

def loss_cross_entropy_softmax(x, y):
    # TO DO
    e_x_sum = 0
    for i in range(len(x)):
        e_x_sum += np.exp(x[i])

    y_hat_i = np.exp(x) / e_x_sum

    l = sum(y * np.log(y_hat_i))
    dl_dy = y_hat_i - y

    return l, dl_dy

def relu(x):
    # TO DO

    y = np.where(x > 0, x, 0)

    return y


def relu_backward(dl_dy, x, y):
    # TO DO
    temp_x = x.reshape(-1)
    temp_dl_dy = dl_dy.reshape(-1)
    dl_dx = np.zeros(temp_dl_dy.shape)

    for i in range(len(temp_dl_dy)):
        if temp_x[i] >= 0:
            dl_dx[i] = temp_dl_dy[i]

    dl_dx = dl_dx.reshape(x.shape)

    return dl_dx

def conv(x, w_conv, b_conv):
    # TO DO

    h, w, c1, c2 = w_conv.shape
    H, W, c1 = x.shape
    directions = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]]
    y = np.zeros((14,14,3))

    for i in range(H):
        for j in range(W):
            for k in range(3):
                cur_filter = w_conv[:, :, :, k]

                cur_sum = 0
                for z in range(len(directions)):
                    dir = directions[z]
                    new_row = i + dir[0]
                    new_col = j + dir[1]

                    if new_row >= 0 and new_col >= 0 and new_row < H and new_col < W:
                        cur_sum += x[new_row][new_col][0] * cur_filter[z // 3][z % 3][0]

                y[i][j][k] = cur_sum

    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):
    # TO DO

    h, w, c1, c2 = w_conv.shape
    H, W, c1 = x.shape
    dl_dw = np.zeros_like(w_conv)
    dl_db = np.zeros_like(b_conv)
    pad = np.pad(x, ((1, 1), (1, 1), (0, 0)))

    for z in range(c2):
        cur_y = dl_dy[:, :, z]
        dl_db[z] = np.sum(cur_y)
        for i in range(h):
            for j in range(w):
                for k in range(c1):
                    cur_x = pad[i: i + H, j:j + W, k]
                    each_pos = np.multiply(cur_y, cur_x)
                    dl_dw[i, j, k, z] = np.sum(each_pos)

    return dl_dw, dl_db

def pool2x2(x):
    # TO DO

    h, w, c = x.shape[0], x.shape[1], x.shape[2]
    y = np.zeros((int(h / 2), int(w / 2), c))

    for i in range(int(h/2)):
        for j in range(int(w/2)):
            for k in range(c):
                y[i][j][k] = np.max(x[i*2:min((i+1)*2,h), j*2:min((j+1)*2,w), k])

    return y

def pool2x2_backward(dl_dy, x, y):
    # TO DO

    dl_dx = np.zeros_like(x)
    h, w, c = x.shape
    for i in range(int(h/2)):
        for j in range(int(w/2)):
            for k in range(c):
                order = np.argmax(x[i * 2 : min((i+1)*2, h), j * 2 : min((j+1)*2, w), k])
                dl_dx[i*2 + order // 2][j*2 + order % 2] = dl_dy[i][j]

    return dl_dx

def flattening(x):
    # TO DO
    y = x.reshape(-1, order = 'F')
    y = y.reshape((len(y), 1))
    return y

def flattening_backward(dl_dy, x, y):
    # TO DO
    dl_dx = dl_dy.reshape(x.shape, order='F')
    return dl_dx

def train_slp_linear(mini_batch_x, mini_batch_y):
    # TO DO

    learning = 0.03
    decay = 0.8
    iteration = 2001
    batch_num = len(mini_batch_x)

    w = np.random.normal(0, 1, size=(10, 196))
    b = np.random.normal(0, 1, size=(10, 1))

    for i in range(1,iteration):
        if i % 1000 == 0:
            learning *= decay

        dL_dw = np.zeros((10, 196))
        dL_db = np.zeros(10)

        cur_index = i % batch_num
        cur_x = mini_batch_x[cur_index]
        cur_y = mini_batch_y[cur_index]

        for j in range(len(cur_x[0])):

            x = cur_x[:, j]
            y = cur_y[:, j]

            y_tilde = fc(x.reshape(196, 1), w, b)
            l, dl_dy = loss_euclidean(y_tilde.reshape(-1), y)
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y_tilde)

            dL_dw += dl_dw
            dL_db += dl_db

        w -= learning/len(cur_x[0]) * dL_dw
        b -= learning/len(cur_x[0]) * dL_db.reshape((10,1))

    return w, b

def train_slp(mini_batch_x, mini_batch_y):
    # TO DO
    learning = 0.2
    decay = 0.9
    iteration = 2001
    batch_num = len(mini_batch_x)

    w = np.random.normal(0, 1, size=(10, 196))
    b = np.random.normal(0, 1, size=(10, 1))

    for i in range(1, iteration):
        if i % 1000 == 0:
            learning *= decay

        dL_dw = np.zeros((10, 196))
        dL_db = np.zeros(10)

        cur_index = i % batch_num
        cur_x = mini_batch_x[cur_index]
        cur_y = mini_batch_y[cur_index]

        for j in range(len(cur_x[0])):
            x = cur_x[:, j]
            y = cur_y[:, j]

            y_tilde = fc(x.reshape(196, 1), w, b)
            l, dl_dy = loss_cross_entropy_softmax(y_tilde.reshape(-1), y)
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y_tilde)

            dL_dw += dl_dw
            dL_db += dl_db

        w -= learning / len(cur_x[0]) * dL_dw
        b -= learning / len(cur_x[0]) * dL_db.reshape((10, 1))

    return w, b

def train_mlp(mini_batch_x, mini_batch_y):
    # TO DO

    learning = 0.23
    decay = 0.9
    iteration = 2500
    batch_num = len(mini_batch_x)

    w1 = np.random.normal(0, 1, size=(30, 196))
    b1 = np.random.normal(0, 1, size=(30, 1))
    w2 = np.random.normal(0, 1, size=(10, 30))
    b2 = np.random.normal(0, 1, size=(10, 1))

    for i in range(1, iteration):
        if i % 1000 == 0:
            learning *= decay

        dL_dw1 = np.zeros((30, 196))
        dL_db1 = np.zeros(30)
        dL_dw2 = np.zeros((10, 30))
        dL_db2 = np.zeros(10)

        cur_index = i % batch_num
        cur_x = mini_batch_x[cur_index]
        cur_y = mini_batch_y[cur_index]

        for j in range(len(cur_x[0])):
            x = cur_x[:, j]
            y = cur_y[:, j]

            y_tilde1 = fc(x.reshape(196, 1), w1, b1)
            y_tilde1 = relu(y_tilde1)
            y_tilde2 = fc(y_tilde1.reshape(30,1), w2, b2)

            l, dl_dy = loss_cross_entropy_softmax(y_tilde2.reshape(-1), y)

            dl_dx, dl_dw2, dl_db2 = fc_backward(dl_dy, y_tilde1, w2, b2, y_tilde2)
            dl_dx = relu_backward(dl_dx, y_tilde1, y_tilde1)
            dl_dx, dl_dw1, dl_db1 = fc_backward(dl_dx.reshape(-1), x, w1, b1, y_tilde1)

            dL_dw1 += dl_dw1
            dL_db1 += dl_db1
            dL_dw2 += dl_dw2
            dL_db2 += dl_db2

        w1 -= learning / len(cur_x[0]) * dL_dw1
        b1 -= learning / len(cur_x[0]) * dL_db1.reshape((30, 1))
        w2 -= learning / len(cur_x[0]) * dL_dw2
        b2 -= learning / len(cur_x[0]) * dL_db2.reshape((10, 1))

    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y):
    # TO DO
    learning = 0.28
    decay = 0.9
    iteration = 2200

    batch_num = len(mini_batch_x)
    # batch_num = math.ceil(len(mini_batch_x) // 2)
    # mini_batch_x = mini_batch_x[:batch_num + 1]
    # mini_batch_y = mini_batch_y[:batch_num + 1]

    w_conv = np.random.normal(0, 1, size=(3, 3, 1, 3))
    b_conv = np.random.normal(0, 1, size= 3)
    w_fc = np.random.normal(0, 1, size=(10, 147))
    b_fc = np.random.normal(0, 1, size=(10, 1))

    for i in range(1, iteration):
        if i % 1000 == 0:
            learning *= decay

        dL_dw_conv = np.zeros((3,3,1,3))
        dL_db_conv = np.zeros(3)
        dL_dw_fc = np.zeros((10, 147))
        dL_db_fc = np.zeros(10)

        cur_index = i % batch_num
        cur_x = mini_batch_x[cur_index]
        cur_y = mini_batch_y[cur_index]

        for j in range(len(cur_x[0])):
            x = cur_x[:, j].reshape((14, 14, 1),order='F')
            y = cur_y[:, j]

            conv1 = conv(x, w_conv, b_conv)
            relu_conv1 = relu(conv1)
            pool_conv1 = pool2x2(relu_conv1)
            flat_conv1 = flattening(pool_conv1)
            fc_conv1 = fc(flat_conv1, w_fc, b_fc)

            l, dl_dy = loss_cross_entropy_softmax(fc_conv1.reshape(-1), y)

            dl_dy, dl_dw_fc, dl_db_fc = fc_backward(dl_dy, flat_conv1, w_fc, 0, 0)
            dl_dy = flattening_backward(dl_dy, pool_conv1, flat_conv1)
            dl_dy = pool2x2_backward(dl_dy, relu_conv1, pool_conv1)
            dl_dy = relu_backward(dl_dy, conv1, relu_conv1)
            dl_dw_conv, dl_db_conv = conv_backward(dl_dy, x, w_conv, b_conv, conv1)

            dL_dw_conv += dl_dw_conv
            dL_db_conv += dl_db_conv
            dL_dw_fc += dl_dw_fc
            dL_db_fc += dl_db_fc

        w_conv -= learning / len(cur_x[0]) * dL_dw_conv
        b_conv -= learning / len(cur_x[0]) * dL_db_conv.reshape(3)
        w_fc -= learning / len(cur_x[0]) * dL_dw_fc
        b_fc -= learning / len(cur_x[0]) * dL_db_fc.reshape((10, 1))

    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    # main.main_slp_linear()
    # main.main_slp()
    main.main_mlp()
    #main.main_cnn()



