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


if __name__ == '__main__':
    test()

