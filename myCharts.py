# -*- coding: UTF-8 -*-
import numpy as np
from logTime import log_time
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
from honda_test import load_df

plt.rc("font",family="SimHei",size="12")

@log_time
def test():
    user_profile = load_df("user_profile")
    # user_profile = user_profile.fillna(0.1)
    # print(user_profile.columns.values.tolist())
    # sb.countplot(data=user_profile, y=user_profile.columns.values.tolist())
    # sb.barplot(data=user_profile, x=user_profile.columns.values.tolist())
    user = user_profile['8ce922c9236a4451bb504ce5ab079244'].dropna(how="all")
    plt.bar(user.index, user.values)
    plt.title("test", fontsize=15)
    plt.grid(True)
    plt.xticks(rotation=90)
    plt.show()


def test2():
    user_profile = load_df("user_profile")
    # user_profile = user_profile.fillna(0.1)
    # print(user_profile.columns.values.tolist())
    # sb.countplot(data=user_profile, y=user_profile.columns.values.tolist())
    # sb.barplot(data=user_profile, x=user_profile.columns.values.tolist())
    user = user_profile['8ce922c9236a4451bb504ce5ab079244']
    print(user_profile.value_counts())
    plt.pie(user.value_counts(), labels=user.value_counts().index, startangle=90,
            counterclock=False, wedgeprops={'width': 0.4})
    plt.title("test", fontsize=15)
    plt.grid(True)
    plt.xticks(rotation=90)
    plt.show()


def test3():
    user_profile = load_df("user_profile")
    # user_profile = user_profile.fillna(0.1)
    # print(user_profile.columns.values.tolist())
    # sb.countplot(data=user_profile, y=user_profile.columns.values.tolist())
    # sb.barplot(data=user_profile, x=user_profile.columns.values.tolist())
    user = user_profile['8ce922c9236a4451bb504ce5ab079244']
    plt.figure(figsize = [10, 5]) # larger figure size for subplots

    # example of somewhat too-large bin size
    plt.subplot(1, 2, 1) # 1 row, 2 cols, subplot 1
    bin_edges = np.arange(0, user_profile['8ce922c9236a4451bb504ce5ab079244'].max()+4, 4)
    plt.hist(data=user_profile, x='8ce922c9236a4451bb504ce5ab079244')

    # example of somewhat too-small bin size
    # plt.subplot(1, 2, 2) # 1 row, 2 cols, subplot 2
    bin_edges = np.arange(0, user_profile['8ce922c9236a4451bb504ce5ab079244'].max()+1/4, 1/4)
    plt.hist(data=user_profile, x='8ce922c9236a4451bb504ce5ab079244')

    plt.show()


if __name__ == '__main__':
    test3()

