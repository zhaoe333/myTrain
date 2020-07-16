# -*- coding: UTF-8 -*-
import numpy as np
from logTime import log_time
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
from honda_test import load_df
import scipy.special

# plt.rc("font",family="SimHei",size="12")


@log_time
def test():
    # print(scipy.special.expit(np.array([0, 1, -1])))
    data = np.array([0, 1, -1])
    d_exp = np.exp(data)
    # print(np.divide(d_exp, np.sum(d_exp)))
    print(data*data)


def f(p):
    return 0.2 * p[0] ** 2 + p[1] ** 2


def gradient1(p):
    h = 1e-6
    x = p[0]
    y = p[1]
    dx = (f([x+h/2, y]) - f([x-h/2, y]))/h
    dy = (f([x, y+h/2]) - f([x, y-h/2]))/h
    return dx, dy


def gradient2(p):
    return 0.4*p[0], 2*p[1]


if __name__ == '__main__':
    p=[3,4]
    print(gradient1(p))
    print(gradient2(p))

