# -*- coding: UTF-8 -*-
import numpy as np
from logTime import log_time
import pandas as pd
import torch
from torchvision import datasets, transforms
import requests
import time
from lxml import etree
import codecs

def load_data():
    vin_df = pd.read_csv("c:\\files\\vin.csv", index_col=0, usecols=[0,1,2,3])
    vin_df['vin_long'] = vin_df['vin_long'].str[3:7]
    print(vin_df[vin_df['Brand'] == 'BBA'].drop_duplicates().sort_values(by=['eseries','vin_long']))


def init_df():
    return pd.DataFrame(columns=['vin', '厂家', '品牌', '车型', '年份', '变速器', '底盘号', '驱动方式', '停产年份'])


def append(df, vin, factory, brand,model,year,speed_changer,chassis,driver,stop_year):
    return df.append({'vin': vin, '厂家': factory, '品牌': brand,
                      '车型': model, '年份': year, '变速器': speed_changer,
                      '底盘号': chassis,'驱动方式': driver, '停产年份': stop_year}, ignore_index=True)


def do_convert():
    df = init_df()
    # 读取数据
    vin_df = pd.read_csv("c:\\files\\vin.csv", index_col=0, usecols=[0, 1, 2, 3])
    vin_df['vin2'] = vin_df['vin_long'].str[3:7]
    for vin in vin_df.loc[vin_df.drop_duplicates(subset=['vin2', 'eseries']).index]['vin_long']:
        try:
            car_info = get_car_info(convert(vin))
            print(vin, *car_info)
            df = append(df, vin, *car_info)
        except:
            print('error:'+vin)
            continue
        time.sleep(2)
    df.to_csv("c:\\files\\vin_convert.csv", encoding='gbk')


def convert(vin):
    print(vin)
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Cookie':'PHPSESSID=vmb86pcics83br7ggb491jet65; relv=1; vin_cookie=0065; Hm_lvt_6c1a81e7deb77ce536977738372f872a=1594696042; BAIDU_SSP_lcr=https://www.baidu.com/link?url=_AbSRyfXvtr44WiopqXwZFp1_KU-xWO7JTwjqd6TGpmUuSPvGaTITzhl-hSRjAv22iEaXxoD6QPLj692Jydqkq&wd=&eqid=f309f62700027543000000025f0d2163; ck_gg_1=y; Hm_lpvt_6c1a81e7deb77ce536977738372f872a=1594699511',
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'
    }
    req_url = "http://www.fenco.cn/Index/search.html?word="+vin
    # req_data = {'leftvin': vin[:8], 'rightvin': vin[9:], 'textfield3': '', 'x':97, 'y':30}
    result = requests.get(req_url)
    return result.text


def get_car_info(html):
    # 建立html的树
    tree = etree.HTML(html)
    # 设置目标路径（标题）
    path_title = '/html/body//table[@class="table table-bordered table-hover"]'
    # 提取节点
    table = tree.xpath(path_title)[0]
    return table.xpath('tr/td//text()')


if __name__ == '__main__':
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
    do_convert()



