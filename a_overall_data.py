"""
数据的类型： 类别， 日期， 数值，ID
数据的分布：每个特征下的数据分布
数据之间的关系：correlation数据的类型： 类别， 日期， 数值，ID
数据的分布：每个特征下的数据分布
数据之间的关系：correlation
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def general_describe(df):
    """
    general describe about the data
    :param df: data frame type
    :return:
    """
    df.head()
    features = df.columns.tolist()
    print('the number of features is ', len(features))
    print('the features are \n', np.array(features))
    print(df.dtypes)


def plot_hist(df, name=''):
    """
    visualize the dataset hist data, if want to dip into the specific data ,use df['feature'].hist()
    :param df:
    :return:
    """
    df.hist(xlabelsize=7, ylabelsize=7, figsize=(12, 10))
    plt.savefig('./pic/'+name+'hist.png')
    plt.show()


def correlation_map(df, name =''):
    """
    correlation map for total data
    notice: the df can only be number no text. so some unique type of values should be transformed first
    :param df: data frame data
    :return:
    """
    colormap = plt.cm.RdBu
    plt.figure(figsize=(14, 12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(df.astype(float).corr(), linewidths=0.1, vmax=1.0,
                square=True, cmap=colormap, linecolor='white', annot=True)
    plt.savefig('./pic/' + name + 'correlation_map.png')
    plt.show()


if __name__ == '__main__':
    filename = './data/under_sample_data.csv'
    df = pd.read_csv(filename)
    general_describe(df)
    plot_hist(df)
    correlation_map(df)
