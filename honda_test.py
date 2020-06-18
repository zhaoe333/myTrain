# -*- coding: UTF-8 -*-
import numpy as np
from logTime import log_time
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
import json,os,math
from sklearn.metrics.pairwise import cosine_similarity


def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def load_data(filename):
    with open("C:\\files\\本田\\data\\"+filename, "r+") as file:
        return json.loads(file.readline())


def handle_user_profile():
    user_profile = load_data('user_profile')
    user_profile_dic = {}
    for user_id in user_profile:
        user_profile_dic[user_id] = pd.Series(data=list(user_profile[user_id].values()), index=user_profile[user_id].keys())
    user_profile_df = pd.DataFrame(user_profile_dic)
    # print(user_profile_df[user_id].dropna())
    # print(user_profile_df.max())

def handle_item_profile():
    item_profile = load_data('item_profile')
    item_profile_dic = {}
    for item_id in item_profile:
        item_profile_dic[item_id] = pd.Series(data=list(item_profile[item_id].values()), index=item_profile[item_id].keys())
    item_profile_df = pd.DataFrame(item_profile_dic)
    # item_profile_df.to_excel(file_path+"item_profile.xls")
    print(item_profile_df[item_index].dropna(axis=0, how='all'))


def to_df(dic_obj):
    obj = {}
    for f_id in dic_obj:
        obj[f_id] = pd.Series(data=list(dic_obj[f_id].values()), index=dic_obj[f_id].keys())
    return pd.DataFrame(obj)


def test():
    # recom_user_item_dic_a = load_data('recom_user_item_dic_a')
    # similar_user_article_dic = load_data('similar_user_article_dic')
    # similar_article_dic = load_data('similar_article_dic')
    # recom_user_item_dic_b = load_data('recom_user_item_dic_b')
    # recom_user_item_dic_c = load_data('recom_user_item_dic_c')
    # item_profile = load_data('item_profile')
    # user_item_score_dic = load_data('user_item_score_dic')
    # similar_user_dic = load_data('similar_user_dic')
    user_profile = load_data('user_profile')
    # user_train_data_set = load_data("user_train_data_set")
    user_profile_df = to_df(user_profile)
    user_profile_df = user_profile_df.fillna(0.0)
    print(type(user_profile_df['8ce922c9236a4451bb504ce5ab079244']))
    # result = user_profile_df.corr(method=my_cos)
    # print(result['8ce922c9236a4451bb504ce5ab079244']['83bec269c25d4cd38a4de8820ca78480'])
    # user_a = user_profile_df['83bec269c25d4cd38a4de8820ca78480']
    # user_b = user_profile_df['8ce922c9236a4451bb504ce5ab079244']
    # print(df['8ce922c9236a4451bb504ce5ab079244'].dropna())
    # print(df.idxmax()['8ce922c9236a4451bb504ce5ab079244'])
    # df.to_csv(file_path + "user_profile.csv", sep="\t")
    # df.to_excel(file_path + "user_profile.xls")


def my_cos(a, b):
    cos = 0.0
    cos_a = 0.0
    cos_b = 0.0
    for i in range(len(a)):
        cos += a[i] * b[i]
        cos_a += a[i] ** 2
        cos_b += b[i] ** 2
    return cos/math.sqrt(cos_a * cos_b)

def export_excel(filename):
    json_data = load_data(filename)
    df = to_df(json_data)
    df.to_excel(file_path + filename + ".xls")


if __name__ == '__main__':
    user_id = "8ce922c9236a4451bb504ce5ab079244"
    file_path = "C:\\files\\本田\\data\\"
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
    # export_excel("similar_user_dic")
    test()

