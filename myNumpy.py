# -*- coding: UTF-8 -*-
import numpy as np
from logTime import log_time
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb


@log_time
def test(test_matrix):
    print(sum(test_matrix)/len(test_matrix))


@log_time
def test2(test_matrix):
    print(np.mean(test_matrix))


def test3():
    df = pd.DataFrame({
        'itemA': pd.Series(data=[1,2,5],index=['a','b','c']),
        'itemB': pd.Series(data=[4,5,6], index = ['b','c','d'])
    })
    print(df[df['itemA']>1])
    # print(df.isna())
    # print(np.mean(df))
    # print(np.median(df.dropna(axis=0)))
    # print(df.corr())
    # print(df.dropna(axis=0))
    # print(df.dropna(axis=1))


if __name__ == '__main__':
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
    test3()

