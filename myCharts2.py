# -*- coding: UTF-8 -*-
import numpy as np
from logTime import log_time
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
from honda_test import load_df

# plt.rc("font",family="SimHei",size="12")


@log_time
def test():
    fuel_econ = load_fuel_econ()
    base_color = sb.color_palette()[0]
    # print(fuel_econ.head())
    plt.subplot(1, 2, 1)
    sb.violinplot(data=fuel_econ, x='VClass', y='comb', color=base_color, inner=None)

    plt.subplot(1, 2, 2)
    sb.boxplot(data=fuel_econ, x='VClass', y='comb')
    plt.grid(True)
    plt.xticks(rotation=15)
    plt.show()


def test2():
    fuel_econ = load_fuel_econ()
    plt.subplot(1, 2, 1)
    sb.violinplot(data=fuel_econ, x='VClass', y='comb', inner=None)
    plt.xticks(rotation=45)
    car_classes = ['Minicompact Cars', 'Subcompact Cars', 'Compact Cars', 'Midsize Cars', 'Large Cars']
    fuel_econ['VClass'] = fuel_econ['VClass'].astype(pd.api.types.CategoricalDtype(ordered=True, categories=car_classes))
    print(fuel_econ['VClass'])
    plt.subplot(1, 2, 2)
    sb.violinplot(data=fuel_econ, x='VClass', y='comb', inner=None)
    plt.xticks(rotation=45)
    plt.show()


def test3():
    fuel_econ = load_fuel_econ()
    fuel_econ['trans'] = fuel_econ['trans'].apply(lambda x: x.split(' ')[0])
    group_result = fuel_econ.groupby(['VClass', 'trans']).size()
    group_result = group_result.reset_index(name='count')
    group_result = group_result.pivot(index='VClass', columns='trans', values='count')
    print(type(group_result))
    print(group_result.head())
    plt.subplot(1, 2, 1)
    sb.heatmap(data=group_result, annot=True, fmt='d')
    plt.subplot(1, 2, 2)
    sb.countplot(data=fuel_econ, x='VClass', hue='trans')
    plt.xticks(rotation=45)
    plt.show()


def load_pokemon():
    return pd.read_csv('C:\\files\\AIPND-master\\Matplotlib\\data\\pokemon.csv')


def load_fuel_econ():
    return pd.read_csv('C:\\files\\AIPND-master\\Matplotlib\\data\\fuel_econ.csv')


if __name__ == '__main__':
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
    test3()

