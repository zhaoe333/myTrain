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


if __name__ == '__main__':
    test()

