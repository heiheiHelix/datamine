# -*- coding: UTF-8 -*-
__author__ = 'jennyzhang'

import pandas as pd
import json
import sys
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier

file = 'wine/winemag-data_first150k.csv'
log_path = 'log.txt'
f = open(log_path,'w',encoding="utf-8")
sys.stdout= f

#打开csv
def open_csv(file):
    try:
        data = pd.read_csv(file)
        return data
    except Exception:
        print('cannot open ', file)

#获取数据
def init():
    #打开excel表格
    data = open_csv(file)
    # #获得Sheet1的行数与列数
    rows,cols = data.shape  #行数,列数
    wine_cate = data.columns #每个属性
   
    #获取标称属性
    nominalCol=['country','designation','province','region_1','region_2','winery']
    
    #获取数值属性
    chemicalCol=['points', 'price']


    return data,nominalCol,chemicalCol



#求出标称量的频数
def nominalDataFrequency(nominalCol,data):
    for col_name in nominalCol:
        dataCol = data[col_name]
        dicts = {}
        for item in dataCol:
            if item in dicts: 
                dicts[item] += 1
            else:
                dicts[item] = 1
        # print('``````````````````````````````')
        # print(col_name)
        # print(dicts)

def checkMissingValues(data):
    na = data.isna().sum()
    return na#.to_dict()



#数值属性，给出最大、最小、均值、中位数、四分位数及缺失值的个数
def statistic(chemicalCol,data):
    #参数统计
    dicts = {}
    for col_name in chemicalCol:
        fiveNumber = {}
        dataCol = data[col_name]
        fiveNumber["Min"] = dataCol.min()
        fiveNumber["Q1"] = dataCol.quantile(q=0.25)
        fiveNumber["Median"] = dataCol.median()
        fiveNumber["Q3"] = dataCol.quantile(q=0.75)
        fiveNumber["Max"] = dataCol.max()
        naDict = checkMissingValues(dataCol)
        dicts[col_name] = [fiveNumber,{'naDict:':naDict}]
    print(dicts)


def histograph(data,x_label,log_flag=True):#,x_label,y_label,title):
    
    # 绘制直方图
    plt.hist(x = data, # 指定绘图数据
            bins = 20, # 指定直方图中条块的个数
            log=log_flag,
            color = 'steelblue', # 指定直方图的填充色
            edgecolor = 'black' # 指定直方图的边框色
            )
    # 添加x轴和y轴标签
    plt.xlabel(x_label)
    plt.ylabel('frequence')
    # 添加标题
    plt.title('wine')
    # 显示图形
    plt.savefig('{}.jpg'.format(x_label))


def N_Sgraph(data,x_label):
    # 构造正态分布的列表数组
    data = data.dropna(inplace=False)
    plt.boxplot(data, notch=False, vert=True)
    plt.title('box plot')
    plt.xlabel(x_label)
    # plt.show()
    plt.savefig('{}_box.jpg'.format(x_label))

def delete_missing(data):
    data_1 = data.copy()
    data_1 = data_1.dropna(axis=0)
    return data_1

def replace_mode_missing(data,chemicalCol):
    data_2 = data.copy()
    for col_name in chemicalCol:
        dataCol = data_2[col_name]
        data_2[col_name] = dataCol.fillna(dataCol.mode())
    return data_2


def set_missing_prices(data):

    data_lost3 = data.copy()

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    price_df = data_lost3[['price', 'points']]

    # 乘客分成已知年龄和未知年龄两部分
    known_price = price_df[price_df.price.notnull()].values
    unknown_price = price_df[price_df.price.isnull()].values

    # y即目标年龄
    y = known_price[:, 0]

    # X即特征属性值
    X = known_price[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedprice = rfr.predict(unknown_price[:, 1:])
    #     print predictedAges
    # 用得到的预测结果填补原缺失数据
    data_lost3.loc[ (data_lost3.price.isnull()), 'price' ] = predictedprice

    return data_lost3

def knn_missing_filled(data):
    k = 3
    dispersed = True
    data_lost4 = data.copy()
    x_train = data_lost4[data_lost4.price.notnull()]['points'].values.reshape(-1,1)
    y_train = data_lost4[data_lost4.price.notnull()]['price'].values.reshape(-1,1)
    print(len(x_train))
    print(len(y_train))
    test = data_lost4[data_lost4.price.isnull()]['points'].values.reshape(-1,1)

    if dispersed:
        clf = KNeighborsClassifier(n_neighbors = k, weights = "distance")
    else:
        clf = KNeighborsRegressor(n_neighbors = k, weights = "distance")
    
    clf.fit(x_train, y_train)

    data_lost4.loc[ (data_lost4.price.isnull()), 'price' ] = clf.predict(test)

    return data_lost4


if __name__=="__main__":
    #打开excel获取数据
    (data,nominalCol,chemicalCol)=init()

    #对标称属性，给出每个可能取值的频数
    # nominalDataFrequency(nominalCol,data)
    #数值属性，给出最大、最小、均值、中位数、四分位数及缺失值的个数
    # statistic(chemicalCol,data)
    # for col in chemicalCol:
    #     # histograph(data[col],col)
        # N_Sgraph(data[col],col)


    # 1.直接去掉缺失值
    # new_drop_1 = delete_missing(data)
    # #可视化
    # for col in chemicalCol:
    #     histograph(new_drop_1[col],'new_drop_1'+col,False)
    #     # N_Sgraph(new_drop_1[col],'new_drop_1'+col)

    # 2.用众数代替
    # new_drop_2 = replace_mode_missing(data,chemicalCol)
    # #可视化
    # for col in chemicalCol:
    #     histograph(new_drop_2[col],'new_drop_2'+col,False)
    #     # N_Sgraph(new_drop_2[col],'new_drop_2'+col)
    
    # 3.相关关系来填补缺失值
    new_drop_3 = set_missing_prices(data)
    #可视化
    col = 'price'
    # histograph(new_drop_3[col],'new_drop_3'+col,False)
    N_Sgraph(new_drop_3[col],'new_drop_3'+col)

    # 4.KNN填补缺失值
    # new_drop_4 = knn_missing_filled(data)
    # histograph(new_drop_4[col],'new_drop_4'+col)
    # N_Sgraph(new_drop_4[col],'new_drop_4'+col)

    

