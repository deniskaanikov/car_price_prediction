# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 20:37:05 2023

@author: daani
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn. preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split


def read_from_csv(path):
    data = pd.read_csv(path)
    return data


def export_description(data):
    description = data.describe(include='all')
    print(description)
    description.to_csv('description.csv')


def price_hist(data):
    print(data['price'])
    prices = np.array(df_main['price'])
    prices = prices[prices < 20000]
    plt.hist(prices, bins=20, range=(5000, 20000))
    prices = np.array(df_main['price'])
    prices = prices[prices > 20000]
    plt.hist(prices, bins=40, range= (20000, 50000))
    plt.xlabel("Car's price, $")
    plt.ylabel('Count')
    plt.title('The relationship of the price of a car with the number of sales')
    plt.legend(loc=2)
    plt.show()


def top_expensive(data, count):
    name = data.groupby("CarName", as_index=False)["price"].mean()
    name = name.sort_values("price").reset_index(drop=True)
    name = name[len(name.index)-count:]

    fig, ax = plt.subplots(figsize=(10, 20))
    ax.barh(range(len(name.index)), name["price"],align='edge', height=.5, color = 'Indigo')
    ax.set_yticks(range(len(name.index)))
    ax.set_yticklabels(name["CarName"])
    plt.ylim(0, len(name.index))
    plt.xlabel("Car's price, $")
    plt.ylabel('Name of car')
    plt.title('Distribution of prices by different car models')
    plt.show()


def top_cheap(data, count):
    name = data.groupby("CarName", as_index=False)["price"].mean()
    name = name.sort_values("price").reset_index(drop=True)
    name = name[:count]

    fig, ax = plt.subplots(figsize=(10, 20))
    ax.barh(range(len(name.index)), name["price"], align='edge', height=.5, color = 'MediumSeaGreen')
    ax.set_yticks(range(len(name.index)))
    ax.set_yticklabels(name["CarName"])
    plt.ylim(0, len(name.index))
    plt.xlabel("Car's price, $")
    plt.ylabel('Name of car')
    plt.title('Distribution of prices by different car models')
    plt.show()


def link_scatter(data, x, y):
    plt.scatter(data[x], data[y])
    plt.show()


def percent_lower (data, column, value):
    return np.mean(np.array(data[column]) < value)*100


def percent_higher (data, column, value):
    return np.mean(np.array(data[column]) > value)*100


def data_gaps(data):
    gaps = data.isnull().sum()
    return gaps


def data_duplicated(data):
    duplicats = data[data.duplicated()]
    return duplicats


def index_outliers(data, column):
    q75, q25 = data[column].quantile(0.75), data[column].quantile(0.25)
    iqr = q75 - q25
    outlier_index = data[(data[column] > q75 + 1.5*iqr) | (data[column] < q25 - 1.5*iqr)].index
    return outlier_index


def info_outliers(data, column):
    return data[(data[i] > data[i].mean() + 3 * data[i].std()) | (data[i] < data[i].mean() - 3 * data[i].std())]


def del_all_outliers(data):
    for i in num_columns:
        data = data[(data[i] <= data[i].mean() + 3 * data[i].std()) & (data[i] >= data[i].mean() - 3 * data[i].std())]
    return data


def transform (data):
    for column in cat_columns:
        if (data[column].dtype != 'float64') & (data[column].dtype != 'int64'):
            enc = LabelEncoder()
            enc.fit(data[column])
            data[column] = enc.transform(data[column])
    return data


def linreg (x_train, x_test, y_train):
    reg = LinearRegression()
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)
    print(reg.intercept_, reg.coef_)
    return y_pred


def KNN(x_train, x_test, y_train, y_test, n):
    reg = KNeighborsRegressor(n_neighbors=n)
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)
    return y_pred


def dec_tree(x_train, x_test, y_train, y_test):
    reg = DecisionTreeRegressor(min_samples_leaf=7, min_samples_split=7, criterion='squared_error')
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)
    return y_pred


def RF(x_train, x_test, y_train, y_test):
    reg = RandomForestRegressor(min_samples_leaf=7, min_samples_split=7, criterion='squared_error')
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)
    return y_pred

def verif (y_pred, y_test, name):
    print('___________', name, '____________')
    print('r2 %s'%(r2_score(y_pred, y_test)))
    print('MAPE %s'%(mean_absolute_percentage_error(y_pred, y_test)))



#MAIN

#импорт файла с датафреймом и начальная обработка
df_main = read_from_csv(r'C:\Users\daani\Downloads\archive\CarPrice_Assignment.csv')
df_main = df_main.drop('car_ID', axis=1)
df_main['symboling'] = df_main['symboling'].astype('object')
print (df_main)
print (df_main.info())

#разделение колонок со строковыми и числовыми данными
cat_columns = df_main.select_dtypes(include=['object']).columns.tolist()
num_columns = df_main.select_dtypes(include=['float64', 'int64']).columns.tolist()

export_description(df_main)

#общий график всех цен
price_hist(df_main)

#графики топ дорогих и дешевых
top_expensive(df_main, 15)
top_cheap(df_main, 15)

#сколько авто стоят дешевле 20000
print (percent_lower(df_main, column = 'price', value = 20000))

print (data_gaps(df_main))

print (data_duplicated(df_main))

#cat_columns.remove('CarName')
df_main = transform(df_main)
print(df_main.info())

#изучение корреляции
df_num = df_main[num_columns]
sns.heatmap(df_num.corr().round(2), annot=True, cbar=False)
plt.show()

#изучение выбранных параметров на предмет выбросов
link_scatter(df_main, x = 'horsepower', y = 'price')
link_scatter(df_main, x = 'carwidth', y = 'price')
link_scatter(df_main, x = 'curbweight', y = 'price')
link_scatter(df_main, x = 'highwaympg', y = 'price')

#общее число выбросов по всем колонкам методом трех сигм
print("Число выбросов")
for i in num_columns:
    print(i, info_outliers(df_main, i).shape[0])

#очистка от выбросов
df_main = del_all_outliers(df_main)
print(df_main)

#показатели в моделях
x_train, x_test, y_train, y_test = train_test_split(df_main[['horsepower', 'carwidth', 'curbweight', 'highwaympg']], df_main[df_main.columns[-1]], random_state=2)
verif(linreg (x_train, x_test, y_train), y_test, 'linear regression')

data_columns = cat_columns + num_columns
df_edit = df_main[data_columns]
x_train, x_test, y_train, y_test = train_test_split(df_edit[['CarName', 'drivewheel', 'horsepower', 'carwidth', 'curbweight', 'highwaympg']], df_main[df_main.columns[-1]],
                                                     random_state=2)
verif(KNN(x_train, x_test, y_train, y_test, 5), y_test, 'KNN')
verif(dec_tree(x_train, x_test, y_train, y_test), y_test, 'Decision tree')
verif(RF(x_train, x_test, y_train, y_test), y_test, 'Random forest')