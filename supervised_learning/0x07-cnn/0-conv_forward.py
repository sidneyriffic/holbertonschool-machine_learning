#!/usr/bin/env python3
"""Convlutional forward propagation, manually"""


import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Convlutional forward propagation, manually"""
    if padding == "valid":
        input = A_prev
    else:
        xpad = int(W.shape[1] / 2)
        ypad = int(W.shape[0] / 2)
        input = np.pad(A_prev, ((0, 0), (ypad - 1 + W.shape[1] % 2, ypad),
                               (xpad - 1 + W.shape[1] % 2, xpad), (0, 0)), 'constant')
    bottom = input.shape[1] - W.shape[0] + 1
    right = input.shape[2] - W.shape[1] + 1
    #print(bottom, right)
    output = np.ndarray((input.shape[0], int((bottom - 1) / stride[0] + 1),
                         int((right - 1) / stride[1] + 1), W.shape[3]))
    y_in = 0
    y_out = 0
    #print(A_prev.shape)
    #print(W.shape)
    #print(input.shape)
    #print(stride)
    #print(output.shape)
    while y_in < bottom:
        x_in = 0
        x_out = 0
        while x_in < right:
            #print("y:", y_in)
            #print("x:", x_in)
            #print(x_in, x_in + W.shape[1])
            #print(y_in, y_in + W.shape[0])
            #print(W[:,:,:, 0], "\n", input[0, y_in:(y_in + W.shape[0]), x_in:(x_in + W.shape[1])])
            mulres = W[np.newaxis, ...] * input[:, y_in:y_in + W.shape[0], x_in:x_in + W.shape[1], :, np.newaxis]
            #print("mulres ", mulres.shape, "\n", mulres[0, ..., 0])
            #print(mulres.sum(axis=(1, 2, 3))[0])
            output[:, y_out, x_out] = activation(mulres.sum(axis=(1, 2, 3)) + b)
            #print("output", output[0, y_out, x_out])
            x_in += stride[1]
            x_out += 1
        y_in += stride[0]
        y_out += 1
    #print(b)
    #print(input.shape)
    #print(output.shape)
    return output
