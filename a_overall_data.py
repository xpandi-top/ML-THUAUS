"""
数据的类型： 类别， 日期， 数值，ID
数据的分布：每个特征下的数据分布
数据之间的关系：correlation数据的类型： 类别， 日期， 数值，ID
数据的分布：每个特征下的数据分布
数据之间的关系：correlation
"""

import pandas as pd
import matplotlib.pyplot as plt

#read file
filename = './data/under_sample_data.csv'
df = pd.read_csv(filename)
df.head()
features = df.columns.tolist()