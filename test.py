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
    x = np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]])
    x2 = np.array([1, 2, 3, 4, 5])
    print(np.sum(x * x2, axis=1))


if __name__ == '__main__':
    test()

